import spacy
import nltk
from time import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from nltk.corpus import verbnet as vn
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import csv


class CorpusReader:
    ### TASK 1 ###
    def load_data(self, filename):
        print("Loading corpus....")
        # loading the file
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            training_sentences = []
            training_labels = []
            for row in reader:
                training_sentences.append(row[0])
                training_labels.append(row[1])
        print("Loading Finished....")
        return training_sentences, training_labels


class Preprocessor:

    def preprocessing(self, training_sentences, training_labels):
        print("Pre-processing training data....")
        # getting sentences from file
        # sentences = nltk.sent_tokenize(training_sentences[0])

        # cleaning each sentence
        for i in range(0, len(training_sentences)):

            # lower case all the letters
            sentence = training_sentences[i].lower()

            # replacing period from the end of sentence
            if sentence[len(sentence) - 1] == '.':
                sentence = sentence[0:len(sentence) - 1]

            # removing comma
            if ',' in sentence:
                sentence = sentence.replace(",", "")

            training_sentences[i] = sentence
        print("Pre-processing completed....")
        return training_sentences, training_labels

    def preprocessing_test(self, training_sentences):
        print("Pre-processing test data....")
        for i in range(0, len(training_sentences)):

            sentence = training_sentences[i]
            # fetching just the sentence
            start = sentence.find('"')
            end = len(sentence) - 1
            sentence = sentence[start + 1:end]

            # lower case all the letters
            sentence = sentence.lower()

            # replacing period from the end of sentence
            if sentence[len(sentence) - 1] == '.':
                sentence = sentence[0:len(sentence) - 1]

            # removing comma
            if ',' in sentence:
                sentence = sentence.replace(",", "")

            training_sentences[i] = sentence
        print("Pre-processing completed....")
        return training_sentences


class FeatureExtractor:
    def __init__(self, ):
        self.train_x = []
        self.train_y = []
        # loading spacy model
        self.nlp = spacy.load("en_core_web_sm")

    def extract_feature(self, training_sentences, training_labels):
        print("Feature extraction initialization....")
        self.train_y = training_labels

        for i in range(0, len(training_sentences)):
            sentence, input_data, e1, e2 = self.get_lexical_features(training_sentences[i])
            input_data = self.get_dependency_features(sentence, input_data, e1, e2)
            input_data = self.get_hypernyms(sentence, input_data, e1, e2)
            input_data = self.get_name_entity(sentence, input_data, e1, e2)
            self.train_x.append(input_data)
        print("Feature extraction completed....")

        return self.train_x, self.train_y

    def get_lexical_features(self, sentence):

        # get the entity e1 and e2
        e1_start = sentence.find("<e1>")
        e1_end = sentence.find("</e1>")
        e2_start = sentence.find("<e2>")
        e2_end = sentence.find("</e2>")
        e1 = sentence[e1_start + 4:e1_end]
        e2 = sentence[e2_start + 4:e2_end]

        ### TASK 2.1 ###
        # tokenizing the sentence
        words_token = nltk.word_tokenize(sentence)
        after_e2 = ""
        before_e1 = ""
        # before_e2 = ""

        ## fetching words before e1 and after e2 ##
        for j in range(0, len(words_token)):
            # word before e1
            if words_token[j - 1] == "<" and words_token[j] == "e1":
                if words_token[0] != "<":
                    before_e1 = words_token[j - 2]
                else:
                    before_e1 = 'NA'
            # word after e2
            elif words_token[j - 2] == "/e2" and words_token[j - 1] == ">":
                if words_token[len(words_token) - 1] != ">":
                    after_e2 = words_token[j]
                else:
                    after_e2 = 'NA'

            ### TASK 2.3 ###
            # POS tags for all words
            words_pos = nltk.pos_tag(words_token)
            e1_pos = ""
            e2_pos = ""

            before_e1_pos = "NA"
            after_e2_pos = "NA"
            # POS tags for entity 1 and entity 2 and words before e1 and after e2
            for k in range(0, len(words_pos)):
                # POS tags for e1
                if words_pos[k][0] == e1:
                    e1_pos = words_pos[k][1]
                # POS tags for e2
                elif words_pos[k][0] == e2:
                    e2_pos = words_pos[k][1]
                # POS tags for word after e2
                if words_pos[k][0] == after_e2:
                    after_e2_pos = words_pos[k][1]
                # POS tags for word before e1
                if words_pos[k][0] == before_e1:
                    before_e1_pos = words_pos[k][1]

        ## fetch words between e1 and e2 ##
        # when e1 comes first in the sentence
        if e1_start < e2_start:
            tmp_words_between = sentence[(e1_end + 5):e2_start].strip()

        # when e2 comes first in the sentence
        else:
            tmp_words_between = sentence[(e2_start + 5):e1_start].strip()

        # Extracting 0 to 10 words between e1 and e2
        words_between = []
        if len(tmp_words_between) != 0:
            # converting in list of words
            tmp_words_between = tmp_words_between.split(" ")
            for j in range(0, 6):
                if j < len(tmp_words_between):
                    words_between.append(tmp_words_between[j])
                else:
                    words_between.append("NA")
        else:
            words_between = ["NA", "NA", "NA", "NA", "NA", "NA"]

        ## POS tagging for 0 to 10 words between e1 and e2 ##
        words_between_pos = []
        if len(tmp_words_between) != 0:
            pos_val = nltk.pos_tag(tmp_words_between)
            for j in range(0, 6):
                if j < len(pos_val):
                    words_between_pos.append(pos_val[j][1])
                else:
                    words_between_pos.append("NA")
        else:
            words_between_pos = ["NA", "NA", "NA", "NA", "NA", "NA"]

        # adding the lexical features
        words_between_string = ""
        words_between_pos_string = ""
        for i in range(0, len(words_between)):
            if i == 0:
                words_between_string = words_between_string + words_between[i]
                words_between_pos_string = words_between_pos_string + words_between_pos[i]
            else:
                words_between_string = words_between_string + " " + words_between[i]
                words_between_pos_string = words_between_pos_string + " " + words_between_pos[i]

        # creating data for training
        input_data = ""
        input_data = before_e1 + " " + before_e1_pos + " " + e1 + " " + e1_pos + " " + words_between_string + " " + words_between_pos_string + " " + e2 + " " + e2_pos + " " + after_e2 + " " + after_e2_pos

        # self.train_x.append(input_data)

        # removing entity annotations
        sentence = sentence.replace("<e1>", '')
        sentence = sentence.replace("</e1>", '')
        sentence = sentence.replace("<e2>", '')
        sentence = sentence.replace("</e2>", '')

        # sentences with no entity annotation
        # train_sentences_no_e.append(sentence)

        ## calling dependency feature extraction function ##
        # self.get_dependency_features(sentence, input_data, e1, e2)
        return sentence, input_data, e1, e2

    def get_lexical_features_test(self, sentence):

        # get the entity e1 and e2
        e1_start = sentence.find("<e1>")
        e1_end = sentence.find("</e1>")
        e2_start = sentence.find("<e2>")
        e2_end = sentence.find("</e2>")
        e1 = sentence[e1_start + 4:e1_end]
        e2 = sentence[e2_start + 4:e2_end]

        ### TASK 2.1 ###
        # tokenizing the sentence
        words_token = nltk.word_tokenize(sentence)
        after_e2 = ""
        before_e1 = ""
        ## fetching words before e1 and after e2 ##
        for j in range(0, len(words_token)):
            # word before e1
            if words_token[j - 1] == "<" and words_token[j] == "e1":
                if words_token[0] != "<":
                    before_e1 = words_token[j - 2]
                else:
                    before_e1 = 'NA'
            # word after e2
            elif words_token[j - 2] == "/e2" and words_token[j - 1] == ">":
                if words_token[len(words_token) - 1] != ">":
                    after_e2 = words_token[j]
                else:
                    after_e2 = 'NA'

            ### TASK 2.3 ###
            # POS tags for all words
            words_pos = nltk.pos_tag(words_token)
            e1_pos = ""
            e2_pos = ""

            before_e1_pos = "NA"
            after_e2_pos = "NA"
            # POS tags for entity 1 and entity 2 and words before e1 and after e2
            for k in range(0, len(words_pos)):
                # POS tags for e1
                if words_pos[k][0] == e1:
                    e1_pos = words_pos[k][1]
                # POS tags for e2
                elif words_pos[k][0] == e2:
                    e2_pos = words_pos[k][1]
                # POS tags for word after e2
                if words_pos[k][0] == after_e2:
                    after_e2_pos = words_pos[k][1]
                # POS tags for word before e1
                if words_pos[k][0] == before_e1:
                    before_e1_pos = words_pos[k][1]

        ## fetch words between e1 and e2 ##
        # when e1 comes first in the sentence
        if e1_start < e2_start:
            tmp_words_between = sentence[(e1_end + 5):e2_start].strip()

        # when e2 comes first in the sentence
        else:
            tmp_words_between = sentence[(e2_start + 5):e1_start].strip()

        # Extracting 0 to 10 words between e1 and e2
        words_between = []
        if len(tmp_words_between) != 0:
            # converting in list of words
            tmp_words_between = tmp_words_between.split(" ")
            for j in range(0, 6):
                if j < len(tmp_words_between):
                    words_between.append(tmp_words_between[j])
                else:
                    words_between.append("NA")
        else:
            words_between = ["NA", "NA", "NA", "NA", "NA", "NA"]

        ## POS tagging for 0 to 10 words between e1 and e2 ##
        words_between_pos = []
        if len(tmp_words_between) != 0:
            pos_val = nltk.pos_tag(tmp_words_between)
            for j in range(0, 6):
                if j < len(pos_val):
                    words_between_pos.append(pos_val[j][1])
                else:
                    words_between_pos.append("NA")
        else:
            words_between_pos = ["NA", "NA", "NA", "NA", "NA", "NA"]

        # adding the lexical features
        words_between_string = ""
        words_between_pos_string = ""
        for i in range(0, len(words_between)):
            if i == 0:
                words_between_string = words_between_string + words_between[i]
                words_between_pos_string = words_between_pos_string + words_between_pos[i]
            else:
                words_between_string = words_between_string + " " + words_between[i]
                words_between_pos_string = words_between_pos_string + " " + words_between_pos[i]

        # words_between_distance = len(words_between)

        # creating data for training
        input_data = ""
        input_data = before_e1 + " " + before_e1_pos + " " + e1 + " " + e1_pos + " " + words_between_string + " " + words_between_pos_string + " " + e2 + " " + e2_pos + " " + after_e2 + " " + after_e2_pos
        # self.train_x.append(input_data)

        return input_data

    def get_dependency_features(self, sentence, input_data, e1, e2):

        # dependency parsing
        doc = self.nlp(sentence)
        dep_e1_root = ""
        dep_e1_root_dependency = ""
        dep_e1_root_pos = ""
        dep_e2_root = ""
        dep_e2_root_dependency = ""
        dep_e2_root_pos = ""

        for chunk in doc.noun_chunks:

            if str(chunk.root.text) == e1:
                dep_e1_root = chunk.root.head.text
                dep_e1_root_dependency = chunk.root.dep_
                dep_e1_root_pos = chunk.root.pos_

            if str(chunk.root.text) == e2:
                dep_e2_root = chunk.root.head.text
                dep_e2_root_dependency = chunk.root.dep_
                dep_e2_root_pos = chunk.root.head.pos_

        dependency_features = dep_e1_root + " " + dep_e1_root_dependency + " " + dep_e1_root_pos + " " + dep_e2_root + " " + dep_e2_root_dependency + " " + dep_e2_root_pos
        input_data = input_data + " " + dependency_features
        return input_data

    def get_hypernyms(self, sentence, input_data, e1, e2):
        e1_hypernym = ""
        e2_hypernym = ""
        e1_hyponym = ""
        e2_hyponym = ""
        e1_meronym = ""
        e2_meronym = ""
        e1_holonym = ""
        e2_holonym = ""
        syn = wordnet.synsets(e1)
        i = 1
        for x in syn:
            if i > 1:
                break
            # extracting hypernyms
            if len(x.hypernyms()) == 0:
                e1_hypernym = "NA"

            else:
                index = x.hypernyms()[0].name().find(".n")
                hyper = x.hypernyms()[0].name()
                hyper = hyper[0:index]
                e1_hypernym = hyper

            # extracting hyponyms
            if len(x.hyponyms()) == 0:
                e1_hyponym = "NA"

            else:
                index = x.hyponyms()[0].name().find(".n")
                hypo = x.hyponyms()[0].name()
                hypo = hypo[0:index]
                e1_hyponym = hypo

            # extracting meronyms
            if len(x.part_meronyms()) == 0:
                e1_meronym = "NA"
            else:
                index = x.part_meronyms()[0].name().find(".n")
                mero = x.part_meronyms()[0].name()
                mero = mero[0:index]
                e1_meronym = mero
            # extracting holonyms
            if len(x.part_holonyms()) == 0:
                e1_holonym = "NA"
            else:
                index = x.part_holonyms()[0].name().find(".n")
                holo = x.part_holonyms()[0].name()
                holo = holo[0:index]
                e1_holonym = holo
            i = i + 1
        if len(syn) == 0:
            e1_hypernym = "NA"
            e1_hyponym = "NA"

        syn = wordnet.synsets(e2)
        i = 1
        for x in syn:
            if i > 1:
                break
            # extracting hypernyms
            if len(x.hypernyms()) == 0:
                e2_hypernym = "NA"
            else:
                index = x.hypernyms()[0].name().find(".n")
                hyper = x.hypernyms()[0].name()
                hyper = hyper[0:index]
                e2_hypernym = hyper

                # extracting hyponyms
                if len(x.hyponyms()) == 0:
                    e2_hyponym = "NA"
                else:
                    index = x.hyponyms()[0].name().find(".n")
                    hypo = x.hyponyms()[0].name()
                    hypo = hypo[0:index]
                    e2_hyponym = hypo

                # extracting meronyms
                if len(x.part_meronyms()) == 0:
                    e1_meronym = "NA"
                else:
                    index = x.part_meronyms()[0].name().find(".n")
                    mero = x.part_meronyms()[0].name()
                    mero = mero[0:index]
                    e2_meronym = mero

                # extracting holonyms
                if len(x.part_holonyms()) == 0:
                    e1_holonym = "NA"
                else:
                    index = x.part_holonyms()[0].name().find(".n")
                    holo = x.part_holonyms()[0].name()
                    holo = holo[0:index]
                    e2_holonym = holo
                i = i + 1

        if len(syn) == 0:
            e1_hypernym = "NA"
            e1_hyponym = "NA"
            e1_meronym = "NA"
            e1_holonym = "NA"

        input_data = input_data + " " + e1_hypernym + " " + e1_hyponym + " " + e1_meronym + " " + e1_holonym + " " + e2_hypernym + " " + e2_hyponym + " " + e2_meronym + " " + e2_holonym
        return input_data

    def get_name_entity(self, sentence, input_data, e1, e2):
        import spacy
        nl = spacy.load("en_core_web_sm")
        doc = nl(sentence)
        e1_ner = "NA"
        e2_ner = "NA"

        for ent in doc.ents:
            if str(ent.text) == e1:
                e1_ner = ent.label_
            if str(ent.text) == e2:
                e2_ner = ent.label_
        input_data = input_data + " " + e1_ner + " " + e2_ner
        return input_data

    def data_vectorization(self, train_x, train_y):
        print("Features vectorization....")
        # encoding labels to number
        Encoder = LabelEncoder()
        train_y = Encoder.fit_transform(train_y)

        ## TF-IDF (transforming string training data to numerical vectors) ##
        # calling the TfidfVectorizer
        vectorize = TfidfVectorizer()
        # fitting the model and passing our sentences right away:
        # train_x = vectorize.fit_transform(train_x)
        train_x_fitted = vectorize.fit(train_x)
        # print("vectorize.vocabulary :", vectorize.vocabulary_)
        # print("vectorize.get_feature_names :", vectorize.get_feature_names())
        # print("vectorize.analyzer :", vectorize.analyzer())
        train_x_transformed = train_x_fitted.transform(train_x)
        print("Features vectorization completed....")
        return train_x_transformed, train_y, vectorize, Encoder


class RelationExtractor:

    def __init__(self):
        self.model = None

    def fit_(self, training_vector, label_vector):
        print("Training model initialization....")
        # Creating an instance of NB classifier
        self.model = naive_bayes.MultinomialNB()
        # self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
        # fitting Naive Bayes
        self.model.fit(training_vector, label_vector)

        # predictions_NB = self.model.predict(training_vector)
        print("Training model completed....")

    def predict(self, test_X, vectorize, encoder, single_test=False):
        # encoding labels to number

        # train_y = Encoder.fit_transform(train_y)

        if single_test:
            # preprocessing test sentences
            Pp = Preprocessor()
            test_X = Pp.preprocessing_test(test_X)

            # extracting features for test sentences
            Fe = FeatureExtractor()
            test_vector = []
            print("Extracting test data Lexical features initialization....")
            for i in range(0, len(test_X)):
                test_vector.append(Fe.get_lexical_features_test(test_X[i]))
            print("Test Data Lexical features Extraction completed....")

            # fitting the model and passing our sentences right away:
            test_x_tfidf = vectorize.transform(test_vector)
        else:
            # fitting the model and passing our sentences right away:
            test_x_tfidf = vectorize.transform(test_X)

        # predict the labels on validation dataset
        predictions_NB = self.model.predict(test_x_tfidf)

        predictions_test = encoder.inverse_transform(predictions_NB)

        return predictions_test

    def accuracy(self, prediction, expected_prediction):
        count = 0

        for i in range(0, len(prediction)):
            if prediction[i] == expected_prediction[i]:
                print("i = ", i, "  ", prediction[i])
                count += 1
        return count / len(prediction)


train_time_start = int(time())
# load data
cr = CorpusReader()
training_data, labels = cr.load_data(r'X_train.csv')

# preprocessing data
pp = Preprocessor()
training_data_preprocessed, training_labels_preprocessed = pp.preprocessing(training_data, labels)

# feature extraction
fe = FeatureExtractor()
training_X, training_Y = fe.extract_feature(training_data_preprocessed, training_labels_preprocessed)

# Encoding text to numerical values
training_X_vectorized, training_Y_vectorized, vectorizer, encoder = fe.data_vectorization(training_X, training_Y)

# fit the training dataset on the classifier
rx = RelationExtractor()
rx.fit_(training_X_vectorized, training_Y_vectorized)
train_time_end = int(time())

# load test data
cr = CorpusReader()
test_data, test_labels = cr.load_data(r'X_test.csv')

test_time_start = int(time())
# preprocessing test data
pp = Preprocessor()
test_data_preprocessed, test_labels_preprocessed = pp.preprocessing(test_data, test_labels)

# feature extraction
fe = FeatureExtractor()
test_X, test_Y = fe.extract_feature(test_data_preprocessed, test_labels_preprocessed)

# Encoding text to numerical values
predictions_train = list(rx.predict(training_X, vectorizer, encoder, False))
predictions_test = list(rx.predict(test_X, vectorizer, encoder, False))

test_time_end = int(time())
unique_label = ["Other",
                "Entity-Destination(e1,e2)", "Cause-Effect(e2,e1)", "Instrument-Agency(e2,e1)",
                "Content-Container(e1,e2)", "Component-Whole(e1,e2)", "Entity-Origin(e1,e2)",
                "Message-Topic(e1,e2)", "Entity-Origin(e2,e1)", "Product-Producer(e2,e1)",
                "Cause-Effect(e1,e2)", "Member-Collection(e2,e1)", "Message-Topic(e2,e1)",
                "Product-Producer(e1,e2)", "Component-Whole(e2,e1)", "Content-Container(e2,e1)",
                "Member-Collection(e1,e2)", "Instrument-Agency(e1,e2)", "Entity-Destination(e2,e1)"]
print()
print()
print("<--------------------------- Training Data Details  :------------------------------------------> ")
print()
print("Total time taken for training : ", train_time_end - train_time_start, " seconds.")

training_results1 = precision_recall_fscore_support(training_Y, predictions_train, average='macro')
training_results2 = precision_recall_fscore_support(training_Y, predictions_train, average=None, labels=unique_label)
training_precision = list(training_results2[0])
training_recall = list(training_results2[1])
training_f_score = list(training_results2[2])
training_count = list(training_results2[3])
print("Classes                       Count")
print("------------------------------------------")
for i in range(0, len(unique_label)):
    extra_space = ""
    if len(unique_label[i]) < 25:
        needed_space = 25 - len(unique_label[i])
        while needed_space > 0:
            extra_space = extra_space + " "
            needed_space = needed_space - 1

    print(unique_label[i], extra_space, "     ", training_count[i])
print()
print("Classes                                      Precision                    Recall                    F-Score")
print("--------------------------------------------------------------------------------------------------------------")
for i in range(0, len(unique_label)):
    extra_space_class = ""
    if len(unique_label[i]) < 25:
        needed_space = 25 - len(unique_label[i])
        while needed_space > 0:
            extra_space_class = extra_space_class + " "
            needed_space = needed_space - 1
    extra_space_precision = ""
    if len(str(training_precision[i])) < 19:
        needed_space = 19 - len(str(training_precision[i]))
        while needed_space > 0:
            extra_space_precision = extra_space_precision + " "
            needed_space = needed_space - 1
    extra_space_recall = ""
    if len(str(training_recall[i])) < 19:
        needed_space = 19 - len(str(training_recall[i]))
        while needed_space > 0:
            extra_space_recall = extra_space_recall + " "
            needed_space = needed_space - 1

    print(unique_label[i], extra_space_class, "     ", training_precision[i], extra_space_precision, "     ",
          training_recall[i], extra_space_recall, "     ", training_f_score[i])
print("-----------------------------------------------------------------------------------------------------------")
extra_space_class = ""
if len("Overall") < 25:
    needed_space = 25 - len("Overall")
    while needed_space > 0:
        extra_space_class = extra_space_class + " "
        needed_space = needed_space - 1
extra_space_precision = ""
if len(str(training_results1[0])) < 19:
    needed_space = 19 - len(str(training_results1[0]))
    while needed_space > 0:
        extra_space_precision = extra_space_precision + " "
        needed_space = needed_space - 1
extra_space_recall = ""
if len(str(training_results1[1])) < 19:
    needed_space = 19 - len(str(training_results1[1]))
    while needed_space > 0:
        extra_space_recall = extra_space_recall + " "
        needed_space = needed_space - 1
print("Overall", extra_space_class, "     ", training_results1[0], extra_space_precision, "     ", training_results1[1],
      extra_space_recall, "     ", training_results1[2])
print("Accuracy Score -> ", rx.accuracy(predictions_train, training_Y) * 100)
print()
print()

print("<--------------------------- Test Data Details  :------------------------------------------> ")
print()
print("Total time taken for testing : ", test_time_end - test_time_start, " seconds.")
test_results1 = precision_recall_fscore_support(test_Y, predictions_test, average='macro')
test_results2 = precision_recall_fscore_support(test_Y, predictions_test, average=None, labels=unique_label)
test_precision = list(test_results2[0])
test_recall = list(test_results2[1])
test_f_score = list(test_results2[2])
test_count = list(test_results2[3])
print("Classes                       Count")
print("------------------------------------------")
for i in range(0, len(unique_label)):
    extra_space = ""
    if len(unique_label[i]) < 25:
        needed_space = 25 - len(unique_label[i])
        while needed_space > 0:
            extra_space = extra_space + " "
            needed_space = needed_space - 1

    print(unique_label[i], extra_space, "     ", test_count[i])
print()
print("Classes                                      Precision                    Recall                    F-Score")
print("--------------------------------------------------------------------------------------------------------------")
for i in range(0, len(unique_label)):
    extra_space_class = ""
    if len(unique_label[i]) < 25:
        needed_space = 25 - len(unique_label[i])
        while needed_space > 0:
            extra_space_class = extra_space_class + " "
            needed_space = needed_space - 1
    extra_space_precision = ""
    if len(str(test_precision[i])) < 19:
        needed_space = 19 - len(str(test_precision[i]))
        while needed_space > 0:
            extra_space_precision = extra_space_precision + " "
            needed_space = needed_space - 1
    extra_space_recall = ""
    if len(str(test_recall[i])) < 19:
        needed_space = 19 - len(str(test_recall[i]))
        while needed_space > 0:
            extra_space_recall = extra_space_recall + " "
            needed_space = needed_space - 1

    print(unique_label[i], extra_space_class, "     ", test_precision[i], extra_space_precision, "     ",
          test_recall[i], extra_space_recall, "     ", test_f_score[i])
print("-----------------------------------------------------------------------------------------------------------")
extra_space_class = ""
if len("Overall") < 25:
    needed_space = 25 - len("Overall")
    while needed_space > 0:
        extra_space_class = extra_space_class + " "
        needed_space = needed_space - 1
extra_space_precision = ""
if len(str(test_results1[0])) < 19:
    needed_space = 19 - len(str(test_results1[0]))
    while needed_space > 0:
        extra_space_precision = extra_space_precision + " "
        needed_space = needed_space - 1
extra_space_recall = ""
if len(str(test_results1[1])) < 19:
    needed_space = 19 - len(str(test_results1[1]))
    while needed_space > 0:
        extra_space_recall = extra_space_recall + " "
        needed_space = needed_space - 1
print("Overall", extra_space_class, "     ", test_results1[0], extra_space_precision, "     ", test_results1[1],
      extra_space_recall, "     ", test_results1[2])

print("Accuracy Score -> ", rx.accuracy(predictions_test, test_Y) * 100)
