
public class Inhabitants
{
	public static void trial() {
	    User emp1=new User("Krishna",1,status.Teacher); 
	    User emp2=new User("Me-Rh",11,status.Student); 
	    
	    User emp3=new User("Radha",22,status.Student); 
	    User emp4=new User("Meera",32,status.Student);
	    
		Hashtable htable = new Hashtable();
		
		htable.insert(emp1);
		htable.insert(emp2);
		htable.insert(emp3);
		htable.insert(emp4);
		
		int idx=1;
		htable.printContents(idx);
		htable.printContents(2);
		
		htable.remove(emp2);
		htable.printContents(idx);
		
		htable.lookUp(1);
	}
	
	public static void main(String[] args) {
		Hashtable htable = new Hashtable();
		//create a thread that takes this as input to its constructor.
		//takes user input when started.
		InputUser[] iu=new InputUser[10];
		
		for(int i=0;i<10;i++) {
			iu[i]=new InputUser(htable);
			iu[i].start();
		}
		
		
		
		
	}
	
}
