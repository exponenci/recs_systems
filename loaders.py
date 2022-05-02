import pandas.io.sql as sqlio
from pandas import read_csv
from functools import reduce
import re
import string


from nltk.stem.snowball import SnowballStemmer


class StemmerWrapper:
    def __init__(self):
        self.stemmer = SnowballStemmer("russian")

    @staticmethod
    def clean_string(sample_s: str) -> str:
        # string to lowercase
        sample_s = sample_s.strip().lower()
        # removing one-symbol words
        sample_s = re.sub(r'\b[ЁёА-я]{1}\b', '', sample_s)
        # removing punctuation
        sample_s = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', sample_s)
        # removing one-digit numbers
        sample_s = re.sub(r'\b[0-9]{1}\b', '', sample_s)
        # replacing several-in-a-row space symbols with only one space
        sample_s = re.sub(r'\s+', ' ', sample_s)
        return sample_s.strip()


class Loader:
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def merge_contents(self,
                       table: str,
                       main_id: str,
                       content_cname: str,
                       columns: list):
        pass

    @staticmethod
    def split_series(series):
        """
        :param series: pd.core.series.Series
        :return: pd.core.series.Series
        """
        # handling None values separately
        return series.apply(lambda x: x.split() if type(x) == str else [])

    def format_columns(self,
                       dataframe,
                       main_id_cname: str,
                       content_cname: str,
                       columns: list):
        # initializing new column in dataframe with empty strings
        dataframe[content_cname] = ''

        # remember items-id-representing column
        id_series = dataframe[main_id_cname]

        # set dataframe to a `columns`-containing table,
        # where all string infos was split into lists
        dataframe = dataframe[[content_cname] + columns].apply(self.split_series)

        # formatting all rows
        dataframe[content_cname] = reduce(  # firstly, we put add all lists to content containing column
                lambda prev, el: prev + dataframe[el],
                columns,
                dataframe[content_cname]
        ).apply(  # then we would stem all words in this column
                lambda iterable: [self.stemmer.stem(w) for w in iterable]
        ).apply(  # lastly, we join lists to string
                lambda iterable: ' '.join(iterable)
        ).apply(
                StemmerWrapper.clean_string
        )

        # set item-representing-column's data
        dataframe[main_id_cname] = id_series

        # return table representing relationship item - formatted_item_content
        return dataframe[[main_id_cname, content_cname]]

    def parse(self, table: str, columns: list):
        pass


class DataBaseLoader(Loader):
    def __init__(self, stemmer, connection):
        super().__init__(stemmer)
        self.connection = connection

    def merge_contents(self,
                       table: str,
                       main_id_cname: str,
                       content_cname: str,
                       columns: list):
        """
        :param table: (str) table name, where from data is going to be read
        :param main_id_cname: (str) `path`-file column-name representing id of items to recommend
                        it is explicit for `path`-file have such column
        :param content_cname: (str) columns containing main content,
                        that will be used to build a content-based model
        :param columns: (list) columns containing main content,
                        that will be used to build a content-based model
        :return: (pd.core.frame.DataFrame) dataframe with 2 columns:
                 1. main_id representing item
                 2. corresponding to main_id column containing formatted `columns` data and
                    named as `content_cname`
        """
        return self.format_columns(
                self.parse(table,
                           columns + [main_id_cname]),
                main_id_cname,
                content_cname,
                columns
        )

    def parse(self, table: str, columns: list):
        sql = "select {0} from {1};".format(
                ','.join(columns),
                table
        )
        return sqlio.read_sql_query(
                sql,
                self.connection
        )


class CsvLoader(Loader):
    def __init__(self, stemmer):
        super().__init__(stemmer)

    def merge_contents(self,
                       path: str,
                       main_id_cname: str,
                       content_cname: str,
                       columns: list):
        """
        :param path: (str) path to csv file, where from data is going to be read
        :param main_id_cname: (str) `path`-file column-name representing id of items to recommend
                        it is explicit for `path`-file have such column
        :param content_cname: (str) columns containing main content,
                        that will be used to build a content-based model
        :param columns: (list) columns containing main content,
                        that will be used to build a content-based model
        :return: (pd.core.frame.DataFrame) dataframe with 2 columns:
                 1. main_id representing item
                 2. corresponding to main_id column containing formatted `columns` data and
                    named as `content_cname`
        """
        return self.format_columns(
                self.parse(path, columns + [main_id_cname]),
                main_id_cname,
                content_cname,
                columns
        )

    def parse(self, table: str, columns: list):
        return read_csv(
                table,
                skipinitialspace=True,
                usecols=columns
        )
