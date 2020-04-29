from collections import defaultdict 
import csv
import ast
import argparse

class Graph: 
    '''
    The code for this class is based on geeksforgeeks.com
    '''
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices 
  
    def addEdge(self, u, v, w): 
        self.graph[u].append([v, w]) 
    
    def topologicalSortUtil(self, v, visited, stack): 
  
        visited[v] = True
  
        for i in self.graph[v]: 
            if visited[i[0]] == False: 
                self.topologicalSortUtil(i[0], visited, stack) 
  
        stack.insert(0,v) 
  
    def topologicalSort(self): 
        visited = [False]*self.V 
        stack =[] 

        for i in range(self.V): 
            if visited[i] == False: 
                self.topologicalSortUtil(i, visited, stack) 
  
        return stack
        
    def isCyclicUtil(self, v, visited, recStack): 
  
        visited[v] = True
        recStack[v] = True
  
        for neighbour in self.graph[v]:
            if visited[neighbour[0]] == False: 
                if self.isCyclicUtil(
                    neighbour[0], visited, recStack) == True: 
                    return True
            elif recStack[neighbour[0]] == True: 
                self.graph[v].remove(neighbour)
                return True
  
        recStack[v] = False
        return False
  
    def isCyclic(self): 
        visited = [False] * self.V 
        recStack = [False] * self.V 
        for node in range(self.V): 
            if visited[node] == False: 
                if self.isCyclicUtil(node, visited, recStack) == True: 
                    return True
        return False

class Stats(object):
    
    def __init__(self):
        self.n_samp = 0
        self.n_sent = 0
        self.n_pair = 0
        self.corr_samp = 0
        self.corr_sent = 0
        self.corr_pair = 0
        self.lcs_seq = 0
        self.tau = 0
        self.dist_window = [1, 2, 3]
        self.min_dist = [0]*len(self.dist_window)
        
    def pairwise_metric(self, g):
        '''
        This  calculates the percentage of skip-bigrams for which the 
        relative order is predicted correctly. Rouge-S metric.
        '''
        common = 0
        for vert in range(g.V):
            to_nodes = g.graph[vert]
            to_nodes = [node[0] for node in to_nodes]
            gold_nodes = list(range(vert+1, g.V))
            common += len(set(gold_nodes).intersection(set(to_nodes)))

        return common
    
    def kendall_tau(self, porder, gorder):
        '''
        It calculates the number of inversions required by the predicted 
        order to reach the correct order.
        '''
        pred_pairs, gold_pairs = [], []
        for i in range(len(porder)):
            for j in range(i+1, len(porder)):
                pred_pairs.append((porder[i], porder[j]))
                gold_pairs.append((gorder[i], gorder[j]))
        common = len(set(pred_pairs).intersection(set(gold_pairs)))
        uncommon = len(gold_pairs) - common
        tau = 1 - (2*(uncommon/len(gold_pairs)))

        return tau
    
    def min_dist_metric(self, porder, gorder):
        '''
        It calculates the displacement of sentences within a given window.
        '''
        count = [0]*len(self.dist_window)
        for i in range(len(porder)):
            pidx = i
            pval = porder[i]
            gidx = gorder.index(pval)
            for w, window in enumerate(self.dist_window):
                if abs(pidx-gidx) <= window:
                    count[w] += 1
        return count
    
    def lcs(self, X , Y): 
        m = len(X) 
        n = len(Y) 

        L = [[None]*(n+1) for i in range(m+1)] 

        for i in range(m+1): 
            for j in range(n+1): 
                if i == 0 or j == 0 : 
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]: 
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j] , L[i][j-1]) 

        return L[m][n] 
    
    def sample_match(self, order, gold_order):
        '''
        It calculates the percentage of samples for which the entire 
        sequence was correctly predicted. (PMR)
        '''
        return order == gold_order
    
    def sentence_match(self, order, gold_order):
        '''
        It measures the percentage of sentences for which their absolute 
        position was correctly predicted. (Acc)
        '''
        return sum([1 for x in range(len(order)) if order[x] == gold_order[x]])
    
    def update_stats(self, nvert, npairs, order, gold_order, g):
        self.n_samp += 1
        self.n_sent += nvert
        self.n_pair += npairs
        
        if self.sample_match(order, gold_order):
            self.corr_samp += 1
        self.corr_sent += self.sentence_match(order, gold_order)
        self.corr_pair += self.pairwise_metric(g)
        self.lcs_seq += self.lcs(order, gold_order)
        self.tau += self.kendall_tau(order, gold_order)
        window_counts = self.min_dist_metric(order, gold_order)
        for w, wc in enumerate(window_counts):
            self.min_dist[w] += wc
        
    def print_stats(self):
        print("Perfect Match: " + str(self.corr_samp*100/self.n_samp))
        print("Sentence Accuracy: " + str(self.corr_sent*100/self.n_sent))
        print("Rouge-S: " + str(self.corr_pair*100/self.n_pair))
        print("LCS: " + str(self.lcs_seq*100/self.n_sent))
        print("Kendall Tau Ratio: " + str(self.tau/self.n_samp))
        for w, window in enumerate(self.dist_window):
            print("Min Dist Metric for window " + str(window) + ": " + \
                                    str(self.min_dist[w]*100/self.n_sent))

def convert_to_graph(data):

    stats = Stats()
    i = 0
    no_docs, no_sents = 0, 0

    while i < len(data):
        ids = data[i][0]

        # get no vertices
        docid, nvert, npairs = ids.split('-')
        docid, nvert, npairs = int(docid), int(nvert), int(npairs)
        
        # create graph obj
        g = Graph(nvert)

        #read pred label
        for j in range(i, i+npairs):
            pred = int(data[j][8])
            log0, log1 = float(data[j][6]), float(data[j][7])
            pos_s1, pos_s2 = int(data[j][4]), int(data[j][5])

            if pred == 0:
                g.addEdge(pos_s2, pos_s1, log0)
            elif pred == 1:
                g.addEdge(pos_s1, pos_s2, log1)          
        
        i += npairs

        while g.isCyclic():
            g.isCyclic()
            
        order = g.topologicalSort()
        no_sents += nvert
        no_docs += 1
        gold_order = list(range(nvert))
        stats.update_stats(nvert, npairs, order, gold_order, g)

        if len(order) != len(gold_order):
            print("yes")
        
    return stats

def readf(filename):
    data = []
    with open(filename, "r") as inp:
        spam = csv.reader(inp, delimiter='\t')
        for row in spam:
            data.append(row)
    return data      

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--file_path", default=None, type=str,
                         required=True, help="The input data dir.")
    args = parser.parse_args()

    data = readf(args.file_path)
    stats = convert_to_graph(data)
    stats.print_stats()

if __name__ == "__main__":
    main()
