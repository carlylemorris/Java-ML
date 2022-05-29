import java.util.*;
import java.io.*;

class NetworkRunner{
   public static void main(String[] args){
   
      Scanner sc = new Scanner(System.in);
      System.out.print("Save file name: ");
      String fname = sc.nextLine();
      Network network;
      try {//Load save file
         FileInputStream fileIn = new FileInputStream("C:/Users/AE/Desktop/JML/saves/"+fname);
         ObjectInputStream in = new ObjectInputStream(fileIn);
         network = (Network)in.readObject();
         in.close();
         fileIn.close();
      } catch (Exception e) {
         System.out.print("Creating save "+fname+" with in shape: ");
         
         String[] shape = sc.nextLine().split(" ",0);
         
         int[] ishape = new int[shape.length];
         for(int i = 0; i<shape.length;i++){
            ishape[i] = Integer.parseInt(shape[i]);
         }
         
         network = new Network(ishape);
         network.write(fname);
      }

      System.out.print("Data file name: ");//Load data file
      String dfname = sc.nextLine();
      Parser p = new Parser(new File("C:/Users/AE/Desktop/JML/data/"+dfname));
      
      //Begin training
      System.out.println("mbs, batches/test, test size, lr: ");
      String[] shps = sc.nextLine().split(" ",0);
      int[] hps = new int[shps.length-1];
      
      for(int i = 0; i<shps.length-1;i++){
         hps[i] = Integer.parseInt(shps[i]);
      }
      
      float lr = Float.parseFloat(shps[3]);
      
      int e = 0;
      System.out.printf("%-10s","tError");
      System.out.printf("%-10s","conAvg");
      System.out.printf("%-10s","conDst");
      System.out.printf("%-10s","outDst");
      System.out.printf("%-10s","oDst^2");
      System.out.printf("%-10s","guessDst");
      System.out.printf("%-10s","gDst^2");
      System.out.printf("%-10s\n","updAvg");
      
      while(true){
      
         double[][] xy = p.next();
         double[][][] gradVectorSum = network.backprop(xy[0],xy[1]);
         
         for(int i = 0; i < hps[0]-1;i++){
            xy = p.next();
            double[][][] gradVector = network.backprop(xy[0],xy[1]);
            
            for(int a = 0;a<gradVector.length;a++){
               for(int b = 0;b<gradVector[a].length;b++){
                  for(int c = 0;c<gradVector[a][b].length;c++){
                     gradVectorSum[a][b][c] = gradVectorSum[a][b][c] + gradVector[a][b][c];
                  }
               }
            }
         }
         
         for(int a = 0;a<gradVectorSum.length;a++){
            for(int b = 0;b<gradVectorSum[a].length;b++){
               for(int c = 0;c<gradVectorSum[a][b].length;c++){
                  gradVectorSum[a][b][c] = gradVectorSum[a][b][c]/hps[0];
               }
            }
         }
         
         network.upd(gradVectorSum,lr);//Actually gradVectorAverage now
         
         
         if(e%hps[1] == 0){
            double[][] fxs = new double[hps[2]][];
            double[][] ys = new double[hps[2]][];
            
            double sumMSE = 0;
            for(int i = 0; i<fxs.length;i++){
               double[][] point = p.next();
               fxs[i] = network.feedforward(point[0]);
               ys[i] = point[1];
               
               sumMSE = sumMSE + Network.MSE(fxs[i],ys[i]);
            }
            sumMSE = sumMSE/hps[2];
         
         
            System.out.printf("%9f ",sumMSE);
            print8(avg(certOf(fxs)));
            print8(stdDev(certOf(fxs)));
            print8(avg(devOf(fxs)));
            print8(stdDev(devOf(fxs)));
            print8(avg(devOf(round(fxs))));
            print8(stdDev(devOf(round(fxs))));
            print8(avg3(gradVectorSum));
            System.out.print("\n");
            
         }
          
         e++;
      }
   }
   
   
   /** certOf
   *@param guesses - matrix of network outputs
   *@return certV - certanty vector
   */
   public static double[] certOf(double[][] guesses){
   double[] certV = new double[guesses[0].length];
   
      for(int c = 0;c<guesses[0].length;c++){
         
         double sum = 0;
         for(int r = 0;r<guesses.length;r++){
            sum = sum + Math.abs(guesses[r][c]-0.5) * 2;
         }
         certV[c] = sum/guesses.length;
      }
      
      return certV;
   }
   
   /** devOf
   *@param guesses - matrix of network outputs
   *@return devV - standard deviation vector
   */
   public static double[] devOf(double[][] guesses){
   double[] devV = new double[guesses[0].length];
      
      for(int c = 0;c<guesses[0].length;c++){//find mean
         
         double sum = 0;
         for(int r = 0;r<guesses.length;r++){
            sum = sum + guesses[r][c];
         }
         devV[c] = sum/guesses.length;
      }
      
      for(int c = 0;c<guesses[0].length;c++){//find standard dev
         
         double sum = 0;
         for(int r = 0;r<guesses.length;r++){
            sum = Math.pow(devV[c]-guesses[r][c],2);
         }
         devV[c] = Math.sqrt(sum/(guesses.length-1));
      }
     
      return devV;
   }
   
   public static double avg3(double[][][] gradVector){
      double out = 0;
      int count = 0;
      for(int a = 0;a<gradVector.length;a++){
            for(int b = 0;b<gradVector[a].length;b++){
               for(int c = 0;c<gradVector[a][b].length;c++){
                  out = out + Math.abs(gradVector[a][b][c]);
                  count++;
               }
           }
      }
      
      return out/count;
   }
   
   public static double avg(double[] a){
      double sum = 0;
      int count = 0;
      for(double s:a){
            sum = sum + s;
            count++;
      }
      
      return sum/count;
   }
   
   public static double stdDev(double[] d){
      double sum = 0;
      double avg = avg(d);
      
      for(int i = 0;i<d.length;i++){
         sum = Math.pow(avg-d[i],2);
      }
      return Math.sqrt(sum/(d.length-1));
   }
   
   public static double[][] round(double[][] m){
      double[][] out = new double[m.length][m[0].length];
      
      for(int i = 0;i < m.length;i++){
         for(int j = 0;j<m[0].length;j++){
            out[i][j] = (double)((int)(m[i][j]+.5));
         }
      }
      
      return out;
   }
   
   public static void print8(double d){
      System.out.printf("%9f ",d);
   }
   
   public static void printM(double[][]m){
      for(double[] row:m){
         System.out.println(Arrays.toString(row));
      }
   }
   
}

