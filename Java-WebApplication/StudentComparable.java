import java.io.*;
import java.lang.*;
import java.util.*;


public class StudentComparable  implements Comparable<StudentComparable>{
	public String name;
	public int rollno;
	public int marks;
	
	public StudentComparable(String n,int rn,int marks) {
		name=n;
		rollno=rn;
		this.marks=marks;
	}
	
	@Override
	public int compareTo(StudentComparable other) {
		//this can be made to compare any two entities of this same class.
		return this.rollno - other.rollno;
	}
	
	@Override
	public String toString() {
		return this.name+ " "+this.rollno+ " " +this.marks+" "+this.hashCode();
	}

	public static void main(String[] args) {
		List<StudentComparable> li = new ArrayList<StudentComparable>();
		li.add(new StudentComparable("Krishna",1,100));
		li.add(new StudentComparable("Sravika",2,99));
		li.add(new StudentComparable("He",3,99));
		
		System.out.println("Unsorted..");
		for(int i=0;i<li.size();i++) {
			System.out.println(li.get(i));
		}
		
		Collections.sort(li);
		System.out.println("Sorted by rollno..");
		for(int i=0;i<li.size();i++) {
			System.out.println(li.get(i));
		}
		
	}
}
