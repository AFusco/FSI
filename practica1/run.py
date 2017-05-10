# Search methods

import search
import time

ab = search.GPSProblem('A', 'B', search.romania)



def run(search, problem):
    total_time = -time.clock()
    final_node = search(problem)
    total_time += time.clock()
    print search.func_name
    if hasattr(final_node, 'visits'):
        print 'Total expanded nodes: ', final_node.visits
    path = final_node.path()
    print path
    print 'Path cost:', final_node.path_cost
    print 'Path length:', len(path)
    print 'Total time (ms):', total_time * 1000
    print


run(search.branch_and_bound_tree_search, ab)
run(search.branch_and_bound_graph_search, ab)
run(search.astar_search, ab)
run(search.branch_and_bound_graph_search_with_underestimation, ab)
run(search.branch_and_bound_tree_search_with_underestimation, ab)
run(search.breadth_first_graph_search, ab)
run(search.depth_first_graph_search, ab)
#run(search.iterative_deepening_search, ab)
#run(search.depth_limited_search, ab)

#print search.astar_search, ab

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
