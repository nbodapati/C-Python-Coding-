public class Node implements print{
	int key;
	String str;
	Node next;
	
	public Node() {
		key=-1;
		str=null;
		next=null;
	}
	
	public Node(int k, String s) {
		key=k;
		str=s;
		next=null;
	}
	
	public Node(final Node l) {
		this(l.key,l.str);  //calling the default constructor.
		System.out.println("Copy constructor called..");
		
	}
	
	public void copy(Node l) {
		l.print_user();
		this.key=l.key;
		this.str=l.str;
		this.next=null;
	}
	
	public void copy(User l) {
		this.key=l.key;
		this.str=l.str;
		this.next=null;
	}
	
	
	public int get_id() {
		 return 0;
		 //do nothing;
	};
	public void print_user() {
		System.out.print(this.key+"  ");
		System.out.println(this.str);
	}
}