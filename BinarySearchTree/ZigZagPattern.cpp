#include<iostream>
//#include<bits/stdc++.h>
#include<stack>
using namespace std;

struct tree_node{
  int value;
  struct tree_node* right;
  struct tree_node* left;

  tree_node(int v=0){
    value=v;
    right=NULL;
    left=NULL;
  }
};

void print_zigzag(int level,stack<tree_node*>& st,stack<tree_node*>& st2){
      int num;    
      if(level%2==0)
       { 
        num=st.size();
       }
      else{
        num = st2.size();}
        
      if(num==0)
        return;
         
       while(num--){
         tree_node* el;     
         if(level%2==0)
         {
           el= st.top();
           st.pop();
           if(el!=NULL){
            cout<<el->value<<"\t";
              if(el->left!=NULL){
                 st2.push(el->left);
                  }
              if(el->right!=NULL){
                 st2.push(el->right);
                 }           
            }
         }
         else{
           el= st2.top();
           st2.pop();
           if(el!=NULL){
            cout<<el->value<<"\t";
                if(el->right!=NULL)
                  st.push(el->right);
                if(el->left!=NULL)
                  st.push(el->left);
            }
         }
       }
           cout<<endl;  
           print_zigzag(level+1,st,st2);
}

int main(){
    tree_node* root=new struct tree_node(1);
    root->left=new tree_node(2);
    root->right=new tree_node(3);

    root->left->left=new tree_node(4);
    root->left->right=new tree_node(5);
    root->right->left=new tree_node(6);
    root->right->right=new tree_node(7);
   
    root->left->left->left=new tree_node(8);
    root->left->left->right=new tree_node(9);
 
    int level=0;
    stack<tree_node*> st;
    stack<tree_node*> st2;
    st.push(root);
    print_zigzag(level,st,st2);

  return 0;
}
