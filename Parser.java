import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

class Parser{
  private Scanner sc;
  File file;
  
  public Parser(File f){
   try{
      sc = new Scanner(f);
   }
   catch(FileNotFoundException e){
      System.out.println(e);
   }
   
   file = f; 
  }
  
  public double[][] next(){
   
   String line;
   
   if(sc.hasNext()){
   line = sc.nextLine();
   }else{
      try{
         sc = new Scanner(file);
      }
      catch(FileNotFoundException e){
         System.out.println(e);
      }
      line = sc.nextLine();
   }
   
   double[][] out = new double[2][];
   
   char[] linea = line.toCharArray();
   
   out[0] = new double[line.indexOf(",")];
   for(int i = 0;i < line.indexOf(",");i++){
      out[0][i] = (int)linea[i]-48;
   }
   
   out[1] = new double[linea.length-line.indexOf(",")-1];
   for(int i = line.indexOf(",")+1;i < linea.length;i++){
      out[1][i-line.indexOf(",")-1] = (int)linea[i]-48;
   }
   
   return out;
  }  
}