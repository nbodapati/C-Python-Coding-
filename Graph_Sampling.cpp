#include<iostream>
#include<cstdio>
#include<vector>
#include<unordered_map>
#include<cmath>
#include<cstdlib> //has rand(),srand()
using namespace std;

class edges{
 public:
  //so all members could be accessed from outside.
  vector<int> edge_end; //this is end point of the edge.
  //can be alinked list as well.
  int count_edges; //number of edges to the source node.

  ostream& operator<<(ostream& os)
{
    //Print number of edges and the list of edges.
    os<<this->count_edges<<'\n';
    for(size_t i=0;i<this->edge_end.size();i++)
       os<<this->edge_end[i]<<"\t";
    os<<endl;

    return os;
}
    ostream& print(){
    cout<<"Num edges"<<this->count_edges<<'\n';
    for(size_t i=0;i<this->edge_end.size();i++)
       cout<<this->edge_end[i]<<"\t";
    cout<<endl;
    return cout;
}

};

class Graph_
{
private:
  bool unfilled; //if True, increment N and M with addition of nodes.
  int N; //Num of nodes in teh graph.
  int M; //Num of edges in the graph.
  unordered_map<int,edges*> hmap;
  vector <int> nodes; //list of nodes in the graph.
  //creates an empty vector - might or might not have allocated space.

public:
  Graph_(){
     unfilled=true;
     N=0;
     M=0;
     //hmap=unordered_map<int,edges*>(); //create an empty map.
     nodes.reserve(11); //create space for 11 values. Never shrink this-only grow.
  }
  Graph_(int n,int m){
    unfilled=false;
    N=n;
    M=m;
    //hmap=unordered_map<int,edges>(); //create an empty map.
    nodes.reserve(11); //create space for 11 values. Never shrink this-only grow.
  }

  //destructor
  ~Graph_(){
    //remove all the edge nodes from hmap. 
      unordered_map<int,edges*>:: iterator itr;
      for(itr = hmap.begin(); itr != hmap.end(); itr++)
      {
         //cout<<"Deleting.."<<itr->first<<endl;
         delete itr->second; 
      }  
    }
 
  int get_num_nodes()
   { 
     return this->nodes.size();
   } 
  void add_nodes_edges(int node,vector<int>m){
       if(unfilled==true)
         {
           N++;
         }
       edges *edges_list =new edges();
       edges_list->edge_end=m;
       edges_list->count_edges=m.size(); //number of edges contained in the vector.
       nodes.push_back(node);
       hmap[node]=edges_list;
  }

  void print_nodes(){
    cout<<"Nodes"<<endl;
    for (size_t i=0;i<nodes.size();i++)
        printf("%d\t",nodes[i]);
    printf("\n");
  }

  void print_edges(int node=-1){
    cout<<"Edges"<<endl;
    if(node==-1){
      //print all nodes and edges.
      unordered_map<int,edges*>:: iterator itr;
      for(itr = hmap.begin(); itr != hmap.end(); itr++)
      { 
        cout << itr->first <<'\t'<<(itr->second)->print()<<endl;
      }
    }
  }
   
  Graph_& sample_graph(Graph_ &subgraph)
    {
    //take reference to the input 
    //modify in place and return.
    //srand(time(NULL));
    //Dont want same pattern of rand() generation 
    float sampling_prob=float(60)/sqrt(this->M);
    //cout<<"sampling_prob: "<<sampling_prob<<endl;
    //Graph_ subgraph;
  
    for (size_t i=0;i<nodes.size();i++)
    {
      //pick a random value from [0,1]
      //if greater than sampling_prob , get the node into the subgraph.
      //else leave it.
      int node=nodes[i];
      float random=rand()/double(RAND_MAX)+17; //get value between 17 and 18
      //cout<<"Node:"<<node<<"random value:"<<random<<endl;
      if(random >=sampling_prob)
        {
          edges* edge_list=hmap[node];
          subgraph.add_nodes_edges(node,edge_list->edge_end);  
        }
      else{
        continue;
        }      
    }

    return subgraph;
    } 
};

int main(){
Graph_ graph(11,12);

int list_nodes[11]={0,1,2,3,4,5,6,7,8,9,10};
int edges0[3]={3,7,9};
int edges1[2]={2,3};
int edges2[1]={4};
int edges3[3]={0,4,8};
int edges4[4]={2,3,5,6};
int edges5[1]={4};
int edges6[1]={4};
int edges7[2]={0,10};
int edges8[2]={3,10};
int edges9[1]={0};
int edges10[2]={7,8};

vector<int> v; 
for (int edge=0;edge<sizeof(edges0)/sizeof(int);edge++)
    v.push_back(edges0[edge]);

graph.add_nodes_edges(0,v);
graph.print_nodes();
graph.print_edges();

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges1)/sizeof(int);edge++)
    v.push_back(edges1[edge]);

graph.add_nodes_edges(1,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges2)/sizeof(int);edge++)
    v.push_back(edges2[edge]);

graph.add_nodes_edges(2,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges3)/sizeof(int);edge++)
    v.push_back(edges3[edge]);

graph.add_nodes_edges(3,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges4)/sizeof(int);edge++)
    v.push_back(edges4[edge]);

graph.add_nodes_edges(4,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges5)/sizeof(int);edge++)
    v.push_back(edges5[edge]);

graph.add_nodes_edges(5,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges6)/sizeof(int);edge++)
    v.push_back(edges6[edge]);

graph.add_nodes_edges(6,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges7)/sizeof(int);edge++)
    v.push_back(edges7[edge]);

graph.add_nodes_edges(7,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges8)/sizeof(int);edge++)
    v.push_back(edges8[edge]);

graph.add_nodes_edges(8,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges9)/sizeof(int);edge++)
    v.push_back(edges9[edge]);

graph.add_nodes_edges(9,v);

v.clear(); //this will remove all the elements from the vector.
for (int edge=0;edge<sizeof(edges10)/sizeof(int);edge++)
    v.push_back(edges10[edge]);

graph.add_nodes_edges(10,v);
graph.print_nodes();
graph.print_edges();
//now that the graph is created - sample it with a probability and 
//return the new graph 
//with sampled nodes vector and hash table with (v,[Nv,|Nv|])

vector<int> sampled(11); //distribution of selection 

for(size_t repeat=0;repeat<100000;repeat++)
{
Graph_ *subgraph=new Graph_();
graph.sample_graph(*subgraph); 
//(*subgraph).print_nodes();
int num_sampled=(*subgraph).get_num_nodes();
sampled[num_sampled]++;
delete subgraph;
//subgraph.print_edges(); //This is working perfectly.
//each time this repeats - a subgraph is deleted and new one created.
}

for(int i=0;i<11;i++)
   cout<<"num nodes: "<<i<<"Num times sampled:"<<sampled[i]<<endl;

return 0;
}
