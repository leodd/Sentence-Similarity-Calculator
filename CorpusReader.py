class CorpusReader:
    def __init__(self, f, encoding='utf8'):
        self.corpus, self.headers = self.formatted_table_data(
            self.load_text(f, encoding=encoding)
        )

    def data(self):
        return self.corpus

    def available_content(self):
        return self.headers

    @staticmethod
    def load_text(f, encoding=None):
        with open(f, 'r', encoding=encoding) as file:
            return file.read()

    @staticmethod
    def formatted_table_data(s):
        rows = s.splitlines()

        res = dict()

        rows_iter = iter(rows)

        # the first row is the header
        header = next(rows_iter).split('\t')

        # add contents to result dict, and use the first column (id) as key
        for row in rows_iter:
            cells = row.split('\t')
            res[cells[0]] = {
                header[i]: cells[i] for i in range(1, len(header))
            }

        return res, header[1:]
