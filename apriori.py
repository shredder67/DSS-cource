from collections import defaultdict
from itertools import chain, combinations
import numpy as np


def powerset(iterable):
    """
    Function to generate all subsets from a set

    Reference:
        https://stackoverflow.com/a/1482316
    """
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [frozenset(ss) for ss in chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))]


class MyARL:
    
    def __init__(self):
        pass
    
    def _generate_new_combinations(self, itemsets):
        combinations = []
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                if len(itemsets[i].difference(itemsets[j])) == 1:
                    combinations.append(itemsets[i] + itemsets[j])
        return combinations


    def _remove_extra_sets(self, itemset, trans) -> list:
        trans_items = frozenset(np.nonzero(trans).tolist())
        filtered_sets = [s for s in itemset if s.issubset(trans_items)]
        return filtered_sets
     
    def apriori(self, X, min_support=0.5, min_confidence=0.6, labels=None):
        """
        Forms a list of association rules

        Parameters:
            X - 2-dimensional numpy array of one-hot encoded transactions
            min_support - minimum support level, float in range (0.0, 1.0)
            min_confidence - minimum condidence level, float in range (0.0, 1.0)
            labels - array of item names, used to replace indicies in rules representation

        Returns:
            rules - list(tuple) - list of pair tuples like ((...), (...)), where first represents

        Reference: 
            https://loginom.ru/blog/apriori
        """
        # Find frequent item sets (with respect to min_support metric)

        rows_number = X.shape[1]
        one_item_set_support = np.array(np.sum(X, axis=0) / rows_number).reshape(-1)
        item_ids = np.arange(X.shape[1])
        k_itemset = [frozenset(item) for item in item_ids[one_item_set_support >= min_support]]
        itemsets_support = one_item_set_support

        while len(k_itemset[-1]) > 0:
            itemset = self.__generate_new_combinations(k_itemset[-1])
            itemset_counts = defaultdict(int)
            for trans in X:
                itemeset_filtered = self._remove_extra_sets(itemset, trans)
                for s in itemeset_filtered:
                    itemset_counts[s] += 1
            itemset_sup = {k: v / rows_number for k, v in itemset_counts.items() if v / rows_number >= min_support}
            itemsets_support.update(itemset_sup)
            k_itemset.append([s for s in itemset if s in itemset_sup])

        common_itemsets = [*k_itemset]
        
        # Association rule generation
        self.rules = []
        for s in common_itemsets:
            for ss in powerset():
                conf = itemsets_support[s] / itemsets_support[ss]
                if conf > min_confidence:
                    self.rules.append([list(ss), list(s)])
        
        if labels:
            for rule in self.rules:
                for t in rule:
                    t = [labels[i] for i in t]

        return self.rules
