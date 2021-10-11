from nltk.stem import PorterStemmer
from nltk.corpus import stopwords	
from math import sqrt
from math import pi
from math import exp
import pandas as pd
import nltk
import re
import string
import punkt
import math


def training(listTraining):
    listComment = listTraining
    listCommentLower = listTraining
    for indexComment in range(0, len(listComment)):
        sentence = listComment[indexComment].translate(
            str.maketrans('', '', string.punctuation)).lower()
        listCommentLower[indexComment] = sentence

    # CETAK CASE FOLDING
    for comment in range(0, len(listComment)):
        print(str(listCommentLower[comment]))

    print('')
    print('=========================== Tokenization ===========================')
    print('')

    listCommentAfterToken = []
    for comment in listCommentLower:
        tokenWord = nltk.tokenize.word_tokenize(comment)
        listCommentAfterToken.append(tokenWord)

    # CETAK TOKENISASI
    for comment in listCommentAfterToken:
        print(str(comment))

    print('')
    print('=========================== Stopword Removal ===========================')
    print('')

    listStopword = set(stopwords.words('english'))

    listCommentStopwords = []

    for comment in listCommentAfterToken:
        notRemoved = []
        for word in comment:
            if word not in listStopword:
                notRemoved.append(word)
        listCommentStopwords.append(notRemoved)

    # CETAK STOPWORD REMOVAL
    for comment in listCommentStopwords:
        print(str(comment))

    print('')
    print('=========================== Stemminng ===========================')
    print('')

    st = PorterStemmer()

    listCommentStem = []

    for comment in listCommentStopwords:
        listStem = []
        for word in comment:
            listStem.append(st.stem(word))
        listCommentStem.append(listStem)

    # CETAK STEMMING
    for test in listCommentStem:
        print(str(test))

    # #################################################################################
    # 5. TERM

    # MENGGABUNGKAN SEMUA TERM MENJADI SATU DAN MENGHILANGKAN TERM YANG BERSIFAT LEBIH DARI 1
    termsTraining = []
    for indexKomentar in range(0, len(listTraining)):
        for word in listCommentStem[indexKomentar]:
            if word not in termsTraining:
                termsTraining.append(word)

    # MEMBUAT METHOD UNTUK MENGHITUNG JUMLAH SUATU KATA DALAM SUATU DOKUMEN
    # KEPERLUAN RUMUS RAW TF

    def countWord(term, document):
        # documentArray = document.split(" ")
        count = 0
        for word in document:
            if term == word:
                count += 1
        return count

    myTerms = {}
    # print("\nTerm Frequency Weighting")
    for term in termsTraining:
        temp = []
        for indexKomentar in range(0, len(listCommentStem)):
            temp.append(
                countWord(term, listCommentStem[indexKomentar]))
        myTerms[term] = temp

    # IDF

    def pengecekan(wtf):
        count = 0
        for number in wtf:
            if number != 0:
                count += 1
        return count

    idf = []
    index = 0
    for word in termsTraining:
        temp = pengecekan(myTerms[word])
        idft = math.log(len(listTraining)/temp, 10)
        idf.append(idft)
        # print("IDF " + word + " = " + str(idf[index]))
        index += 1

    # (TF-IDF)
    def computeTFIDF():
        tfidf = {}
        index = 0
        for term in termsTraining:
            temp = []
            for indexDocument in range(0, len(listTraining)):
                temp.append(myTerms[term][indexDocument] * idf[index])
            tfidf[word] = temp
            # print("TF-IDF " + word + " = " + str(tfidf[word]))
            index += 1

    computeTFIDF()

    

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(x, mean, stdev):
	    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	    return (1 / (sqrt(2 * pi) * stdev)) * exponent

    print(calculate_probability(1.0, 1.0, 1.0))

    # inisialisasi total diisi dengan 0 sepanjang dokumen
    # total = []
    # for indexKomentar in range(0, len(listCommentStem)):
    #     total.append(0)

    # for indexKomentar in range(0, len(total)):
    #     for term in termsTraining:
    #         total[indexKomentar] += myTerms[term][indexKomentar]

    # def countSpecificWordInCategory(word, category):
    #     counter = 0
    #     indexDocument = 0
    #     if category == 'Positif':
    #         for tf in myTerms[word]:
    #             if indexDocument < 15:
    #                 counter = counter + tf
    #             indexDocument += 1
    #     elif category == 'Negatif':
    #         for tf in myTerms[word]:
    #             if indexDocument >= 15:
    #                 counter = counter + tf
    #             indexDocument += 1
    #     return counter

    # def countAllWordInCategory(category):
    #     counter = 0
    #     if category == 'Positif':
    #         indexDocument = 0
    #         for totalTiapDokumen in total:
    #             if indexDocument < 15:
    #                 counter = counter + totalTiapDokumen
    #             indexDocument += 1
    #     elif category == 'Negatif':
    #         indexDocument = 0
    #         for totalTiapDokumen in total:
    #             if indexDocument >= 15:
    #                 counter = counter + totalTiapDokumen
    #             indexDocument += 1
    #     return counter

    # def getTotalTerm():
    #     return len(termsTraining)

    # def countConditionalProbablility(word, category):
    #     return (countSpecificWordInCategory(word, category) + 1) / (countAllWordInCategory(category) + getTotalTerm())

    conProbPositive = []
    conProbNegative = []

    for term in termsTraining:
        conProbPositive.append(countConditionalProbablility(term, 'Positif'))
        conProbNegative.append(countConditionalProbablility(term, 'Negatif'))

    myConditionalProbability = {}
    indexKomentar = 0
    for term in termsTraining:
        temp = []
        temp.append(conProbPositive[indexKomentar])
        temp.append(conProbNegative[indexKomentar])
        myConditionalProbability[term] = temp
        indexKomentar += 1

    returnValue = []
    returnValue.append(myConditionalProbability)
    returnValue.append(termsTraining)
    return returnValue
