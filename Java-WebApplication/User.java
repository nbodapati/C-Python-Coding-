import java.lang.String;

enum status{
	Student,
	Teacher
};

interface print{
  static int n=10;
  public void print_user();
  public int get_id();
}

public class User extends Node implements print 
{
//boolean connected; 
//boolean consumes 16 bytes - 8bytes header, 1 byte payload and 7 more bytes 
//of padding for 8-byte memory alignment.
private  String name; 
public  int id;
private  status student_teacher;
byte connected; //8bit

public User(final String name,int id,status st){
this.name=name;
this.id=id;
this.student_teacher=st;
this.connected=0x00;
}

public User(final User u) {
	this(u.name,u.id,u.student_teacher);
}

public void copy(User l) {
	this.name=l.name;
	this.id=l.id;
	this.student_teacher=l.student_teacher;
	this.connected=l.connected;
}

public int get_id() {
	return this.id;
}

public void print_user(){
   System.out.print(this.name+" ");
   System.out.print(this.id+" ");
   System.out.print(this.student_teacher.toString()+" "+this.connected+" "+"\n"); 
  }  

public void FlipState()
{
     byte flip=0x01;       
     this.connected=(byte)(this.connected ^ flip);
}

public static void trial(int... args){
   System.out.println("Number of args to user class.."+args.length);	
   int number_of_users= args[0]; //this gives the first argument to the function.
   while((number_of_users--)>0) {
	   
   User u= new User("Obama",1,status.Student);
   u.print_user(); 

   u.FlipState();
   u.print_user();

   u.FlipState();
   u.print_user();
  }
}
}


