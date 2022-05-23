
def export_text(decision_tree, feature_names=None, class_names=None, spaces=3, decimals=2):
    """
    Method similar to sklearn.tree.export_text, outputs fitted MyDecisionTree instance structure

    Parameters:
    ----------
    decision_tree : MyDecisionTree
        Fited MyDecisionTree instance

    feature_names : list
        List of feature names to replace indices with
    
    class_names : list
        List of label names to replace indices with
    
    spaces : int
        Tabulation size for node depth level
    
    decimals: int
        Formating parameter for decimal numbers output

    Returns:
    ----------
    report : str
        Text summary of decision tree structure (rules in nodes)
    """
    report = ''
    stack = [(decision_tree.root, 1)]
    str_stack = ['']
    while len(stack) > 0:
        cur, depth = stack.pop(-1)
        report += str_stack.pop(-1)

        indent = ('|' + ' '*spaces)*(depth - 1)
        indent += '|' + '-'*spaces

        if cur.predicted_value is not None:
            label = cur.predicted_value if class_names is None else class_names[cur.predicted_value]
            report += indent + f' class: {label}\n'
            continue

        feature_name = cur.feature_index if feature_names is None else feature_names[cur.feature_index]
        if cur.right_child:
            stack.append((cur.right_child, depth + 1))
            str_stack.append(indent + f' {feature_name} >= {cur.value:.{decimals}f}\n')
        if cur.left_child:
            stack.append((cur.left_child, depth + 1))
            str_stack.append(indent + f' {feature_name} < {cur.value:.{decimals}f}\n')
    
    return report


