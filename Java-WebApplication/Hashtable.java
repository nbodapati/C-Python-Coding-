public class Hashtable {
	private int numel;
	private Node[] llist;
	
	//this is executed each time a hash table is created.
	static {
		System.out.println("Hashtable is created!");
	}
	
	public Hashtable() {
	  this(10); 
	  //default numel=10		
	}
	
	public Hashtable(int n) {
		numel=n;
		llist= new User[numel];
		
	}
	
	public void printContents(int idx) {
		//System.out.println("Printing the contents..");
		Node header=llist[idx];
		
		while(header!=null) {
			header.print_user();
		    header=header.next;
		}
	}
	
	public void insert(User l) {
		int k=l.id;
		int get_idx=k%this.numel;
		
		Node llist_= this.llist[get_idx];
	
		if(llist_==null) {
			//this is the first element.
			this.llist[get_idx]=new User(l);
		}
		else {
			while(llist_.next!=null) {
				llist_=llist_.next;
			}
			
			llist_.next=new User(l);
		}
	}
	
	public void remove(User l) {
		int k=l.get_id();
		int get_idx=k%this.numel;
		Node llist_= this.llist[get_idx];
		Node llist2=llist_;
		while(llist_!=null) {
		     if(llist_.get_id()==k) {
		    	 System.out.println("Found the element");
		    	 break; 
		       }
		     llist2=llist_;
		     llist_=llist_.next; 
		     
		}
		
		if(llist_!=null) {
	    	System.out.println("Remove the element."); 
		    Node next=llist_.next;
		    //copy the contents from next to current and delete next.
		    llist2.next=next;
		}
	  }
	 
	public void lookUp(int id) {
		int get_idx=id%this.numel;
		Node llist_= this.llist[get_idx];
		while(llist_!=null) {
		     if(llist_.get_id()==id) {
		    	 System.out.println("Found the element");
		    	 break; 
		       }
		     llist_=llist_.next; 
		     
		}
		if(llist_==null) {
			System.out.println("Element Not Found");
		}
			
	 }
	
  }
