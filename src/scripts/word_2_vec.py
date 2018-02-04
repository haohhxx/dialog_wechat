import word2vec

word2vec.word2phrase(r'..\..\corpus\dialog_datas\sentence_vocbulart.txt'
                     , r'..\..\corpus\dialog_datas\sentence_vocbulart.txt.phrases'
                     , verbose=True)

word2vec.word2vec(r'..\..\corpus\dialog_datas\sentence_vocbulart.txt.phrases'
                  , r'..\..\corpus\dialog_datas\sentence_vocbulart.txt.phrases.bin'
                  , size=128, verbose=True, min_count=0)

# word2vec.word2clusters(r'..\..\corpus\dialog_datas\sentence_vocbulart.txt'
#                        , r'..\..\corpus\dialog_datas\sentence_vocbulart.txt.cluster'
#                        , 128, verbose=True)











