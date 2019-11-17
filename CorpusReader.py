class CorpusReader:
    def __init__(self, f, encoding='utf8'):
        self.data, self.headers = self.formatted_table_data(
            self.load_text(f, encoding=encoding)
        )

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data

    def __str__(self):
        str_list = list()
        for k, item in self.data.items():
            str_list.append(str(k))
            for header in self.headers:
                str_list.append('\n\t{}: {}'.format(header, item[header]))
            str_list.append('\n')
        return ''.join(str_list)

    def ids(self):
        return self.data.keys()

    def available_contents(self):
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
