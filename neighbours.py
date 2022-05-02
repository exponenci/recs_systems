from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


import structural
import loaders


class SubstituteItemRS(structural.ContentBasedFiltering):
    """
    SUBSTITUTE ITEM
    content-based recommendation system
    to find closest item with similar content
    """

    def __init__(self,
                 stemmer,
                 max_features=1e5,
                 stop_words=stopwords.words("russian")):
        super(SubstituteItemRS, self).__init__(stemmer)
        self.transformer = CountVectorizer(max_features=max_features,
                                           stop_words=stop_words)
        self.similarity = None      # similarity matrix

    def build(self):
        raw_data = self.transformer.fit_transform(self.df_data[self.content_cname])
        self.df_data.drop(self.content_cname, axis=1, inplace=True)
        self.similarity = cosine_similarity(raw_data.toarray())

    def find_closest(self,
                     item_id: str,
                     res_n: int = 10):
        item_index = self.df_data[self.df_data[self.item_id_cname] == item_id].index[0]

        cos_distances = self.similarity[item_index]
        items_list = sorted(list(enumerate(cos_distances)), reverse=True, key=lambda x: x[1])[1:res_n + 1]
        indices = [item[0] for item in items_list]

        # return found items ids
        return self.df_data.iloc[indices][self.item_id_cname].values


if __name__ == '__main__':
    ws = loaders.StemmerWrapper()
    neighbour_item_rs = SubstituteItemRS(ws)
    neighbour_item_rs.load(
            "data/product.csv",
            "product_id",
            "content_info",
            ["product_name", "description", "seller"]
    )
    neighbour_item_rs.build()
    m_item_id = neighbour_item_rs.df_data[0]  # item id as it is stored in database
    res_indices = neighbour_item_rs.find_closest(m_item_id)
    print(res_indices)
