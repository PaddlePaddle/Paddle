from paddle.fluid.core import IndexWrapper, TreeIndex
from .builder import TreeIndexBuilder

class Index(object):
    def __init__(self, name):
        self._name = name

class GraphIndex(Index):
    def __init__(self, name):
        super(GraphIndex, self).__init__(name)

class TreeIndex(Index):
    def __init__(self, name):
        super(TreeIndex, self).__init__(name)
        self._tree = None
        self._builder = TreeIndexBuilder(name)
        self._wrapper = IndexWrapper()

    def _init_by_category(self,input_filename, output_filename):
        self._builder.tree_init_by_category(input_filename, output_filename)
        self._init_tree(output_filename)

    def _init_by_kmeans(self, output_filename):
        self._builder.tree_init_by_kmeans(output_filename)
        self._init_tree(output_filename)

    def _init_tree(self, filename):
        self._tree = self._wrapper.insert_tree_index(self._name, filename)

    # # COMMON
    def get_nodes_given_level(self, level, ret_code=False):
        return self._tree.get_nodes_given_level(level, ret_code)

    def get_parent_path(self, id, start_level=-1, ret_code=False):
        return self._tree.get_parent_path(id, start_level, ret_code)

    def tree_height(self):
        return self._tree.height()

    def tree_branch(self):
        return self._tree.branch()

    # # JTM
    # def get_itemset_given_ancestor(self, relation, ancestor):
    #     return self._tree.get_itemset_given_ancestor(relation, ancestor)

    # def get_children_given_ancestor_and_level(self, ancestor, level, ret_code=True):
    #     return self._tree.get_children_given_ancestor_and_level(ancestor, level, ret_code)

    # def get_relation(self, level):
    #     pass

    # def update_relation(self, relation):
    #     pass