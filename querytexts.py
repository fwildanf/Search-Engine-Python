import re
import buildindex
from nltk.stem import PorterStemmer

#input = [file1, file2, ...]
#res = {word: {filename: {pos1, pos2}, ...}, ...}
class Query:

    def __init__(self, filenames):
        self.filenames = filenames
        self.index = buildindex.BuildIndex(self.filenames)
        self.invertedIndex = self.index.totalIndex
        self.regularIndex = self.index.regdex


    def one_word_query(self, word):
        pattern = re.compile('[\W_]+')
        word = pattern.sub(' ',word)
        re.sub(r'[\W_]+','', word)
        word = PorterStemmer().stem(word)        
        if word in self.invertedIndex.keys():
            return self.rankResults([filename for filename in self.invertedIndex[word].keys()], word)
        else:
            return []

    def free_text_query(self, string):
        pattern = re.compile('[\W_]+')
        string = pattern.sub(' ',string)
        result = []
        for word in string.split():
            result += self.one_word_query(word)
        return self.rankResults(list(set(result)), string)

    def make_vectors(self, documents):
        vecs = {}
        for doc in documents:
            docVec = [0]*len(self.index.getUniques())
            for ind, term in enumerate(self.index.getUniques()):
                docVec[ind] = self.index.generateScore(term, doc)
            vecs[doc] = docVec
        return vecs


    def query_vec(self, query):
        pattern = re.compile('[\W_]+')
        query = pattern.sub(' ',query)
        queryls = query.split()
        queryVec = [0]*len(queryls)
        index = 0
        for ind, word in enumerate(queryls):
            queryVec[index] = self.queryFreq(word, query)
            index += 1
        queryidf = [self.index.idf[word] for word in self.index.getUniques()]
        magnitude = pow(sum(map(lambda x: x**2, queryVec)),.5)
        freq = self.termfreq(self.index.getUniques(), query)
        #print('THIS IS THE FREQ')
        tf = [x/magnitude for x in freq]
        final = [tf[i]*queryidf[i] for i in range(len(self.index.getUniques()))]
        #print(len([x for x in queryidf if x != 0]) - len(queryidf))
        return final

    def queryFreq(self, term, query):
        count = 0
        #print(query)
        #print(query.split())
        for word in query.split():
            if word == term:
                count += 1
        return count

    def termfreq(self, terms, query):
        temp = [0]*len(terms)
        for i,term in enumerate(terms):
            temp[i] = self.queryFreq(term, query)
            #print(self.queryFreq(term, query))
        return temp

    def dotProduct(self, doc1, doc2):
        if len(doc1) != len(doc2):
            return 0
        return sum([x*y for x,y in zip(doc1, doc2)])

    def rankResults(self, resultDocs, query):
        vectors = self.make_vectors(resultDocs)
        #print(vectors)
        queryVec = self.query_vec(query)
        #print(queryVec)
        results = [[self.dotProduct(vectors[result], queryVec), result] for result in resultDocs]
        #print(results)
        #results.sort(key=lambda x: x[0], reverse=True)
        
        #print(results)
        #results = [x[1] for x in results]
        return results


q = Query(['apple-iphone-x-review.txt', 'indonesia-courts-neil-bantleman.txt', 'indonesia-setya-novanto-corruption.txt', 'iphone-x-what-to-know.txt', 'justice-league-review-wonder-woman-batman-dc-comics.txt', 'justice-league-wonder-box-office.txt', 'albatrosses-hit-by-fishing-and-climate.txt', 'bizarre-shape-of-interstellar-asteroid.txt', 'drone-maker-makes-hacking-accusations.txt', 'dubai-airshow-why-the-uae-is-probing-space-agriculture.txt', 'uber-and-volvo-strike-deal-for-24,000-self-drive-cars.txt', 'uk-seeks-future-cyber-security-stars.txt'])
x = input("input : ")
docs = q.free_text_query(x)
res = q.rankResults(docs, x)
#print('\n'.join('{}: {}'.format(*k) for k in enumerate(res)))
print(res)