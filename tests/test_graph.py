import unittest
from nevula.core.tensor import Tensor
from nevula.graph.node import Node
from nevula.graph.graph import Graph
from nevula.graph.capture import graph_capture
from nevula.graph.optimizer import ConstantFoldingPass, DeadCodeEliminationPass, GraphOptimizer
from nevula.engine.plan import Planner, ExecutionPlan
from nevula.engine.executor import Executor
from nevula.engine.compile import compile
from nevula.nn.linear import Linear
from nevula.nn.activations import ReLU
from nevula.nn.container import Sequential


class TestComputationalGraph(unittest.TestCase):
    def test_node_creation(self):
        n1 = Node(node_id=0, name="x", op="input", shape=(2, 3))
        self.assertTrue(n1.is_input)
        self.assertFalse(n1.is_constant)
        self.assertFalse(n1.is_op)
        self.assertEqual(n1.shape, (2, 3))

        n2 = Node(node_id=1, name="c", op="constant", value=5.0)
        self.assertTrue(n2.is_constant)

        n3 = Node(node_id=2, name="add_0", op="add", inputs=[n1, n2])
        self.assertTrue(n3.is_op)
        self.assertEqual(len(n3.inputs), 2)
        self.assertIn(n3, n1.outputs)
        self.assertIn(n3, n2.outputs)

    def test_topological_sort(self):
        graph = Graph()
        x = graph.create_input("x", shape=(1,))
        w = graph.create_input("w", shape=(1,))
        mul = graph.create_node("mul", inputs=[x, w], shape=(1,))
        b = graph.create_input("b", shape=(1,))
        add = graph.create_node("add", inputs=[mul, b], shape=(1,))
        graph.outputs = [add]

        sorted_nodes = graph.topological_sort()
        self.assertEqual(len(sorted_nodes), 5)

        # Ensure dependencies always precede dependent nodes
        indices = {node.id: i for i, node in enumerate(sorted_nodes)}
        self.assertLess(indices[x.id], indices[mul.id])
        self.assertLess(indices[w.id], indices[mul.id])
        self.assertLess(indices[mul.id], indices[add.id])
        self.assertLess(indices[b.id], indices[add.id])

    def test_graph_capture_simple_arithmetic(self):
        x = Tensor([2.0])
        w = Tensor([3.0])
        b = Tensor([1.0])

        with graph_capture() as graph:
            y = x * w
            z = y + b

        self.assertEqual(len(graph.nodes), 5)
        self.assertEqual(len(graph.outputs), 1)

        ops = [n.op for n in graph.nodes]
        self.assertIn("input", ops)
        self.assertIn("mul", ops)
        self.assertIn("add", ops)

        # Output should be the final addition node
        out_node = graph.outputs[0]
        self.assertEqual(out_node.op, "add")

    def test_executor_execution(self):
        x = Tensor([2.0])
        w = Tensor([3.0])
        b = Tensor([1.0])

        with graph_capture() as graph:
            y = x * w
            z = y + b

        executor = Executor()
        result = executor.execute(graph)
        self.assertAlmostEqual(result.item(), 7.0)

        # Re-execute with dynamic feed_dict: x=5, w=4, b=1 -> 5*4 + 1 = 21
        result2 = executor.execute(graph, feed_dict={graph.inputs[0]: Tensor([5.0]), graph.inputs[1]: Tensor([4.0])})
        self.assertAlmostEqual(result2.item(), 21.0)

    def test_constant_folding(self):
        x = Tensor([10.0])
        c1 = Tensor([2.0])
        c2 = Tensor([3.0])

        with graph_capture() as graph:
            # Mark c1 and c2 nodes as constant
            n_c1 = graph.create_constant(c1, name="c1")
            n_c2 = graph.create_constant(c2, name="c2")
            graph._tensor_to_node[id(c1)] = n_c1
            graph._tensor_to_node[id(c2)] = n_c2

            c_sum = c1 + c2
            z = x + c_sum

        # Before folding, there are 2 constants, 1 input x, 1 add for c1+c2, 1 add for x+c_sum = 5 nodes
        self.assertEqual(len(graph.nodes), 5)

        folder = ConstantFoldingPass()
        graph = folder.run(graph)

        # After folding, c1+c2 node should have become a constant with value 5.0
        c_sum_node = graph._tensor_to_node[id(c_sum)]
        self.assertTrue(c_sum_node.is_constant)
        self.assertAlmostEqual(c_sum_node.value.item(), 5.0)

        executor = Executor()
        res = executor.execute(graph)
        self.assertAlmostEqual(res.item(), 15.0)

    def test_dead_code_elimination(self):
        x = Tensor([2.0])
        w = Tensor([3.0])
        b = Tensor([1.0])

        with graph_capture() as graph:
            # Dead operation (never used in final output z)
            dead_val = x - w
            z = x * b

        self.assertIn("sub", [n.op for n in graph.nodes])
        initial_count = len(graph.nodes)

        dce = DeadCodeEliminationPass()
        graph = dce.run(graph)

        # Dead sub operation and any unused nodes must be eliminated
        self.assertNotIn("sub", [n.op for n in graph.nodes])
        self.assertLess(len(graph.nodes), initial_count)

        executor = Executor()
        res = executor.execute(graph)
        self.assertAlmostEqual(res.item(), 2.0)

    def test_planner_and_execution_plan(self):
        x = Tensor([2.0])
        w = Tensor([3.0])
        b = Tensor([1.0])

        with graph_capture() as graph:
            y = x @ w
            z = y + b

        plan = Planner.build_plan(graph)
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(len(plan.steps), len(graph.nodes))

        summary = plan.memory_summary()
        self.assertIn("total_steps", summary)
        self.assertIn("operations", summary)

        rendered = plan.render()
        self.assertIn("Execution Plan:", rendered)
        self.assertIn("Summary:", rendered)

    def test_compile_neural_network(self):
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2),
        )

        compiled_model = compile(model)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])

        eager_out = model(x)
        compiled_out = compiled_model(x)

        self.assertEqual(eager_out.shape, (1, 2))
        self.assertEqual(compiled_out.shape, (1, 2))

        # Output of compiled model must match eager execution exactly
        for o1, o2 in zip(eager_out.data.tolist(), compiled_out.data.tolist()):
            self.assertAlmostEqual(o1, o2, places=5)

        # Re-execute on different batch input
        x2 = Tensor([[0.5, -1.0, 2.0, 0.0]])
        eager_out2 = model(x2)
        compiled_out2 = compiled_model(x2)
        for o1, o2 in zip(eager_out2.data.tolist(), compiled_out2.data.tolist()):
            self.assertAlmostEqual(o1, o2, places=5)


if __name__ == "__main__":
    unittest.main()
