from collections import defaultdict
import pandas as pd
import numpy as np


from itertools import chain, combinations

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
    
    def __generate_new_combinations(self, itemsets):
        combinations = []
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                if len(itemsets[i].difference(itemsets[j])) == 1:
                    combinations.append(itemsets[i] + itemsets[j])
        return combinations

    
    def apriori(self, df, min_support=0.5, min_confidence=0.6):
        """
        Forms a list of association rules

        Parameters:
            df - a pandas DataFrame of one-hot encoded transactions
            min_support - minimum support level, float in range (0.0, 1.0)
            min_confidence - minimum condidence level, float in range (0.0, 1.0)

        Returns:
            rules - list(tuple) - list of pair tuples like ((...), (...)), where first represents

        Reference: 
            https://loginom.ru/blog/apriori
        """
        # Find frequent item sets (with respect to min_support metric)

        X = df.values
        rows_number = X.shape[1]
        one_item_set_support = np.array(np.sum(X, axis=0) / rows_number).reshape(-1)
        item_ids = np.arange(X.shape[1])
        k_itemset = [frozenset(item) for item in item_ids[one_item_set_support >= min_support]]

        while len(k_itemset[-1]) > 0:
            itemset = self.__generate_new_combinations(k_itemset[-1])
            itemset_counts = defaultdict(int)
            for trans in X:
                itemeset_filtered = self._remove_extra_sets(itemset, trans)
                for s in itemeset_filtered:
                    itemset_counts[s] += 1
            k_itemset.append([s for s in itemset if itemset_counts[s] / rows_number >= min_support])

        common_itemsets = [*k_itemset]
        
        # Association rule generation
        # TODO: supp function (actually, need just to safe all calculated confidences itemsets in dict)
        self.rules = []
        for s in common_itemsets:
            for ss in powerset():
                conf = supp(s)/supp(ss)
                if conf > min_confidence:
                    self.rules.append((tuple(ss), tuple(s)))

        return self.rules
