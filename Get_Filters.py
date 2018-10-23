def select_subset(product_df, question, answer):
    """
    function assumes you have already build the answer column

    enter the string corresponding to question number and to answer number
    """
    if answer == 'idk':  # case i don't know the answer return everything
        return (product_df)
    else:
        q_keep = set(product_df.loc[product_df["PropertyDefinitionId"] == float(
            question), "ProductId"].drop_duplicates().index.values)
        a_keep = set(product_df.loc[product_df["answer"] == float(answer), "ProductId"].drop_duplicates().index.values)
        total = a_keep.intersection(q_keep)
        return (product_df.loc[total,])