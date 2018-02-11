f = open(r'../../corpus/dialog_datas/sentence_dialog.txt'
         , 'r', encoding='utf-8')
fw = open(r'../../corpus/dialog_datas/voc', 'w'
          , encoding='utf-8')

wordset = set()
for line in f.readlines():
    sentences = line.split('\t')
    for sentence in sentences:
        for word in sentence.split(' '):
            wordset.add(word)

for i, word in enumerate(wordset):
    fw.write(word + ' ' + str(i) + '\n')
