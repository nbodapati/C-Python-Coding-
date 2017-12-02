//This file takes in as input a 2D array of numbers.
//Computes bag of words representation of them.
//Alternative clustering with min-hashing and
//lsh hashing
#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>
#include<vector>
using namespace std;

int docA[100001];
int docB[100001];
int r_c_mat[10][100001];

int first1_docA;
int first1_docB;

int random_gen()
{
//generate a random number between [0,10]
return rand()%2;
}

//define a function that takes (a,b) as inputs
//returns a lambda function which can be stored
//in a auto variable.
auto function_generator(int a,int b){
   //create a lambda hash function
   int p=100001;
   auto func=[a,b,p](int x)->int{return (a*x+b)%p;};
   return func;
}

int a[10],b[10];

void create_hash_functions()
{
for(int i=0;i<10;i++)
   {
     //int a_=random_gen();
     //int b_=random_gen();
     //cout<<"i= "<<i<<"a= "<<a_<<"b= "<<b_<<endl;
     a[i]=i+1;
     b[i]=1;
   }
}

int* get_hash_functions(int x)
{
int hashed_values[10];
for(int i=0;i<10;i++)
{
  hashed_values[i]=function_generator(a[i],b[i])(x);
}
return hashed_values;
}

void print_contents(int *a,int r,int c)
{
  for(int i=0;i<r;i++)
   {
   for(int j=0;j<c;j++)
      cout<<*((a+i)+j)<<"\t";
   }
    cout<<endl;
}

void create_documents()
{
  int sim=0;
  for(int i=0;i<100001;i++)
   {
     docA[i]=random_gen();
     docB[i]=random_gen();
     //cout<<docA[i]<< docB[i]<<endl;
     if(docA[i]==1 && docB[i]==1)
       sim++;
   }
   cout<<float(sim)/100001<<endl;
}

void walkdown()
{
 for(int i=0;i<100001;i++)
   {
     if(docA[i]==1)
      {
       first1_docA=i;
       break;
      }
   }

 for(int i=0;i<100001;i++)
   {
     if(docB[i]==1)
      {
       first1_docB=i;
       break;
      }
   }
}

float find_similarity(int n_hash)
  {
     int sim_mat[2][n_hash];
     int sim=0;
     for(int i=0;i<n_hash;i++)
        {
            sim_mat[0][i]=r_c_mat[i][first1_docA];
            sim_mat[1][i]=r_c_mat[i][first1_docB];
            cout<<sim_mat[0][i]<<sim_mat[1][i]<<endl;
            if(sim_mat[0][i]== sim_mat[1][i])
               sim++;
        }
  return float(sim)/100001;
  }
int main()
{
  create_documents();
  walkdown();
  create_hash_functions();

  for(int i=0;i<100001;i++)
{
  int *hashed=get_hash_functions(i);
  for(int j=0;j<10;j++)
     r_c_mat[j][i]=hashed[j];
}
  int sim=find_similarity(10);
  printf("%f\n",sim);
  //print_contents(r_c_mat[0],10,100001);
  return 0;
}
