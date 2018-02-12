
public class InputUser extends Thread{
    public Hashtable htable;
    private Thread t;
    public static int num_threads=0;
    public static Object Lock= new Object();
    public String name;
    
    public InputUser(Hashtable ht) {
    	this.htable=ht;
    	num_threads++;
    }
    
    //override this function.
    public  void run() {
    	
    	System.out.println("Running thread.."+this.name);
    	try {
			Thread.sleep(5000);
			synchronized(Lock) {
			System.out.println("Sleeping thread.."+this.name);
			}
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	System.out.println("Exiting thread.."+this.name);
    	
    }
    
    public void start() {
    	this.name="InputUser thread"+num_threads;
    	Thread t= new Thread(this,"InputUser thread"+num_threads);
    	t.run();
    }
}
