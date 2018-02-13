import java.util.*;
import java.lang.*;
import java.io.*;

public class Student {
	public String name;
	public int rollno;
	public int marks;

	public Student(String n,int rn,int marks) {
		name=n;
		rollno=rn;
		this.marks=marks;
	}
	
	@Override
	public String toString() {
		return this.name+ " "+this.rollno+ " " +this.marks;
	}
	
	public static void main_() {
		List<Student> li= new LinkedList<Student>();
		li.add(new Student("Krishna",1,100));
		li.add(new Student("Sravika",2,99));
		li.add(new Student("He",3,99));
		
		System.out.println("Unsorted..");
		for(int i=0;i<li.size();i++) {
			System.out.println(li.get(i));
		}
		
		Collections.sort(li,new sortbyname());
		System.out.println("Sorted by name..");
		for(int i=0;i<li.size();i++) {
			System.out.println(li.get(i));
		}
		
		Collections.sort(li,new sortbymarks());
		System.out.println("Sorted by marks..");
		for(int i=0;i<li.size();i++) {
			System.out.println(li.get(i));
		}
		
		Collections.sort(li,new sortbyrollno());
		System.out.println("Sorted by roll no..");
		for(int i=0;i<li.size();i++) {
			System.out.println(li.get(i));
		}
		
	}
			
}
class sortbyname implements Comparator<Student>{
	  public int compare(Student a, Student b) {
		  return a.name.compareTo(b.name);
	  }
}
class sortbyrollno implements Comparator<Student>{
	  public int compare(Student a, Student b) {
			  return a.rollno - b.rollno;
		}
		  	  
}

class sortbymarks implements Comparator<Student>{
	  public int compare(Student a, Student b) {
			  return a.marks- b.marks;
		}
		  	  
}	



