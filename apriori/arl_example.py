import pandas as pd
from ARL import MyARL

def main():
    # Data preperation
    data = [
        (0, 'Крутой учитель Онидзука'),
        (0, 'Вельзевул'),
        (0, 'Наруто'),
        (1, 'Бродяга'),
        (1, 'Берсерк'),
        (1, 'Атака титанов'),
        (1, 'Наруто'),
        (2, 'ДжоДжо'),
        (2, 'Ван Пис'),
        (2, 'Ванпанчмен'),
        (2, 'Поднятие уровня в одиночку'),
        (2, 'Охотник х Охотник'),
        (2, 'Вельзевул'),
        (3, 'Поднятие уровня в одиночку'),
        (3, 'Ванпанчмен'),
        (3, 'Наруто'),
        (4, 'Поднятие уровня в одиночку'),
        (4, 'Охотник х Охотник'),
        (4, 'Клинок, рассекающий демонов'),
        (4, 'ДжоДжо'),
        (5, 'Ванпанчмен'),
        (5, 'Поднятие уровня в одиночку'),
        (5, 'Кулак Северной звезды'),
        (5, 'Бродяга'),
        (6, 'Крутой учитель Онидзука'),
        (6, 'Вельзевул'),
        (6, 'Бродяга'),
        (7, 'ДжоДжо'),
        (7, 'Д. Грэй-мен'),
        (7, 'Атака титанов'),
        (7, 'Бродяга'),
        (8, 'ДжоДжо'),
        (8, 'Ванпанчмен'),
        (8, 'Поднятие уровня в одиночку'),
        (9, 'Берсерк'),
        (9, 'Охотник х Охотник'),
        (9, 'Бродяга'),
        (9, 'Наруто'),
        (9, 'Вельзевул'),
    ]
    
    user_ids, titles = zip(*data)
    user_ids = list(set(user_ids))
    titles = list(set(titles))

    df = pd.DataFrame(index=user_ids, columns=titles)
    for tr in data:
        uid, title = tr
        df[title][uid] = 1
    df = df.fillna(0)

    # Assosiation rule learning
    arl = MyARL()
    arl.apriori(df.values, min_support=0.15, min_confidence=0.6, labels=df.columns)

    antecedents, consequents, supports, confidences, lifts = zip(*arl.get_rules())
    itemsets, it_supports = zip(*arl.get_popular_itemsets())

    # Formatting and output
    rules_df = pd.DataFrame(data={
        "Antecedent": antecedents,
        "Consequent": consequents,
        "Support": supports,
        "Confidence": confidences,
        "Lift": lifts
    })
    rules_df.index += 1
    rules_df.to_csv('rules.csv')

    popular_itemsts_df = pd.DataFrame(data={
        'Itemset': itemsets,
        'Support': it_supports
    })
    popular_itemsts_df.index += 1
    popular_itemsts_df.to_csv('popular_itemsets.csv')
    


if __name__ == '__main__':
    main()