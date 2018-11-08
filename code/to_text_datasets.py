from sqlalchemy import create_engine
import pandas as pd


# Creates a data frame containing question id's and corresponding text
def question_to_text_df():
    try:
        engine = create_engine('postgresql://dslab:dslab2018@localhost/dslab')
        c = engine.connect()
    except:
        print("Verify connections to database")

    # Get dataframe with question as text from id's
    question_text_df = pd.read_sql_query('''
              SELECT DISTINCT "PropertyDefinition", "PropertyDefinitionId" from product
              WHERE "ProductTypeId"='6'
              ''', c)
    return question_text_df


def answer_to_text_df():
    try:
        engine = create_engine('postgresql://dslab:dslab2018@localhost/dslab')
        c = engine.connect()
    except:
        print("Verify connections to database")

    answer_equivalence_df = pd.read_sql_query('''
              SELECT DISTINCT "PropertyDefinitionOption", "PropertyDefinitionOptionId" from product
              WHERE "ProductTypeId"='6'
              ''', c)
    return answer_equivalence_df
