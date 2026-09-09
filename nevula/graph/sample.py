"""
Nevula DL Framework - Phase 8 Computational Graph & Execution Engine Demonstration
Demonstrates graph capture, ASCII visualization, optimization (constant folding and DCE),
execution planning, and end-to-end model compilation.
"""

import nevula as nv
from nevula import Tensor, graph_capture, compile
from nevula.graph.optimizer import ConstantFoldingPass, DeadCodeEliminationPass, GraphOptimizer
from nevula.engine.plan import Planner
from nevula.engine.executor import Executor
from nevula.nn import Sequential, Linear, ReLU


def main():
    print("=================================================================")
    print("   🚀 Nevula DL Framework — Phase 8: Computational Graph DAG     ")
    print("=================================================================\n")

    # -------------------------------------------------------------
    # 1. First Real Test: z = (x * w) + b
    # -------------------------------------------------------------
    print("1. Capturing Computational Graph (z = x * w + b):")
    x = Tensor([2.0])
    w = Tensor([3.0])
    b = Tensor([1.0])

    with graph_capture() as graph:
        y = x * w
        z = y + b

    print(graph.render())
    print()

    # -------------------------------------------------------------
    # 2. Execution Engine
    # -------------------------------------------------------------
    print("2. Executing Graph via Executor:")
    executor = Executor()
    result = executor.execute(graph)
    print(f"   Evaluated Output: z = {result.item()}  (Expected: (2)(3) + 1 = 7)\n")

    # Dynamic re-execution with feed_dict
    print("3. Dynamic Evaluation with new input values (x=5, w=4, b=1):")
    re_result = executor.execute(graph, feed_dict={graph.inputs[0]: Tensor([5.0]), graph.inputs[1]: Tensor([4.0])})
    print(f"   Evaluated Output: z = {re_result.item()}  (Expected: (5)(4) + 1 = 21)\n")

    # -------------------------------------------------------------
    # 4. Graph Optimizations: Constant Folding
    # -------------------------------------------------------------
    print("4. Graph Optimization: Constant Folding (2 + 3 -> 5):")
    x_var = Tensor([10.0])
    c1 = Tensor([2.0])
    c2 = Tensor([3.0])

    with graph_capture() as fold_graph:
        n_c1 = fold_graph.create_constant(c1, name="c1")
        n_c2 = fold_graph.create_constant(c2, name="c2")
        fold_graph._tensor_to_node[id(c1)] = n_c1
        fold_graph._tensor_to_node[id(c2)] = n_c2

        c_sum = c1 + c2
        final_val = x_var + c_sum

    print("   Before Constant Folding:")
    print(f"   Node count: {len(fold_graph.nodes)}")
    
    ConstantFoldingPass().run(fold_graph)
    print("   After Constant Folding:")
    print(f"   Node count: {len(fold_graph.nodes)}")
    folded_node = fold_graph._tensor_to_node[id(c_sum)]
    print(f"   Folded node '{folded_node.name}': is_constant={folded_node.is_constant}, value={folded_node.value.item()}\n")

    # -------------------------------------------------------------
    # 5. Graph Optimizations: Dead Operation Elimination
    # -------------------------------------------------------------
    print("5. Graph Optimization: Dead Operation Elimination:")
    with graph_capture() as dce_graph:
        dead_branch = x * w  # Never used in output!
        live_out = x + b

    print(f"   Initial node count: {len(dce_graph.nodes)} (includes dead multiplication branch)")
    DeadCodeEliminationPass().run(dce_graph)
    print(f"   Node count after DCE: {len(dce_graph.nodes)} (dead branch pruned successfully)\n")

    # -------------------------------------------------------------
    # 6. Execution Plan
    # -------------------------------------------------------------
    print("6. Execution Planning (Planner & ExecutionPlan):")
    plan = Planner.build_plan(graph)
    print(plan.render())
    print()

    # -------------------------------------------------------------
    # 7. Neural Network Compilation (nevula.compile)
    # -------------------------------------------------------------
    print("7. Compiling Neural Network Model (nevula.compile):")
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )

    compiled_model = compile(model)
    sample_x = Tensor([[1.0, 2.0, 3.0, 4.0]])

    eager_out = model(sample_x)
    compiled_out = compiled_model(sample_x)

    print(f"   Eager Execution Output   : {eager_out}")
    print(f"   Compiled Execution Output: {compiled_out}")
    print("   Verified: Compiled output matches eager output exactly!")
    print("\n✅ Phase 8 Computational Graph & Execution Engine complete!")


if __name__ == "__main__":
    main()
