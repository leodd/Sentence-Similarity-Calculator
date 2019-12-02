from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser, CoreNLPDependencyParser
import os


class CoreNLP:
    def __init__(self, args):
        self.context = dict()
        self.server = None
        self.set_system_env(*args)

    def set_system_env(self, *args):
        idx = 1
        while idx < len(args):
            if args[idx] == '--stanford':
                idx += 1
                standford_path = args[idx]
                self.context['path_to_jar'] = os.path.join(standford_path, 'stanford-corenlp-3.9.2.jar')
                self.context['path_to_models_jar'] = os.path.join(standford_path, 'stanford-corenlp-3.9.2-models.jar')
                print('corenlp jar:', self.context['path_to_jar'])
                print('corenlp models jar:', self.context['path_to_models_jar'])

            elif args[idx] == '--java':
                idx += 1
                java_path = args[idx]
                os.environ['JAVAHOME'] = java_path
                print('java path:', java_path)

            idx += 1

    def start_server(self):
        self.server = CoreNLPServer(**self.context)
        self.server.start()

    def stop_server(self):
        self.server.stop()

    def parse_tree(self, s):
        parser = CoreNLPParser()

        parse = next(parser.raw_parse(s))
        # parse.draw()

        return parse

    def dependency_parse_tree(self, s):
        parser = CoreNLPDependencyParser()

        parse = next(parser.raw_parse(s))

        return parse


if __name__ == '__main__':
    import sys
    corenlp = CoreNLP(sys.argv)
    corenlp.start_server()
    print(corenlp.dependency_parse_tree('aij-Weggen report (A5-0323/2000)'))
    # print(corenlp.parse_tree('aij-Weggen report (A5-0323/2000)'))
    corenlp.stop_server()
