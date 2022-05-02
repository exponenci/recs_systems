import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import structural
import loaders


class SearchEngine(structural.ContentBasedFiltering):
    def __init__(self, stemmer):
        super(SearchEngine, self).__init__(stemmer)
        self.transformer = TfidfVectorizer()
        self.tfidf_matrix = None

    def build(self):
        raw_data = self.transformer.fit_transform(self.df_data[self.content_cname])
        self.tfidf_matrix = pd.DataFrame(raw_data.T.toarray())

    @staticmethod
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self,
               query: str,
               res_n: int = 10):
        rows_count, columns_count = self.tfidf_matrix.shape
        # applying transformer's transform to query and getting it's vector
        query_raw = self.transformer.transform([loaders.StemmerWrapper.clean_string(query)])
        query_vec = query_raw.toarray().reshape(rows_count)

        # counting cosine similarity of query and each item-content (in tfidf_matrix)
        """
            I have tried this code, but it was slower for 10% compared to raw iterations,
            `sim = self.tfidf_matrix.apply(lambda x: self.cosine_sim(x, query_vec))`
        """
        similarities = list()
        for i in range(columns_count):
            similarities.append(
                    self.cosine_sim(
                            self.tfidf_matrix.loc[:, i].values,
                            query_vec
                    )
            )

        # sort from the best matching to the worst
        similarities = sorted(
                enumerate(similarities),
                key=lambda x: x[1],
                reverse=True
        )

        # getting top `res_count` results' indices
        indices = [pair[0] for pair in similarities[:res_n]]

        # return found items ids
        return self.df_data.iloc[indices][self.item_id_cname].values


if __name__ == '__main__':
    ws = loaders.StemmerWrapper()
    search_eng = SearchEngine(ws)
    search_eng.load(
            "data/product.csv",
            "product_id",
            "content_info",
            ["product_name", "description", "seller"]
    )
    search_eng.build()
    res_indices = search_eng.search("серебряное с бриллиантами")
    print(res_indices)
    # print(df_data.iloc[indices]["tags"].values)
