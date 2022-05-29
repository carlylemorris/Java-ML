import java.util.*;
import java.io.*;

class Network implements java.io.Serializable{
   public double[][][] weights;
   public double[][] biases;
   
   //Intitialization Constructor
   public Network(int[] arch){
      Random random = new Random();
      
      weights = new double[arch.length - 1][][];
      biases = new double[arch.length - 1][];
      
      //Initialize array structure
      for(int i = 0;i<arch.length-1;i++){
         weights[i] = new double[arch[i]][arch[i+1]];
         biases[i] = new double[arch[i+1]];
      }
      
      //Initialize weights and biases 
      for(double[][] weight:weights){
       
         for(int r = 0;r<weight.length;r++){
            for(int c = 0;c<weight[0].length;c++){
               weight[r][c] = random.nextGaussian()/Math.sqrt(weight.length);
            }
         }
      }
      
      for(double[] bias:biases){
     
         for(int i = 0;i<bias.length;i++){
            bias[i] = 0;
         }
      }
   }
   
   //Static Methods
   public static double sigmoid(double x){
      return 1/(1+Math.exp(x*-1));
   } 
   
   public static double sigmoidPrime(double x){
      return sigmoid(x)*(1-sigmoid(x));
   }
   
   public static double relu(double x){
      if(x>0){return x;}else{return 0;}
   } 
 
   public static double reluPrime(double x){
      if(x>0){return 1;}else{return 0;}
   }
   
   public static double[] getZ(double[] a,double[][] w,double[] b){
      
      double[] z = new double[b.length];
      for(int c = 0; c<w[0].length; c++){
      
         z[c] = b[c];
         for(int r = 0;r<w.length;r++){
            z[c] = z[c] + a[r] * w[r][c];
         }
      }
      
      return z;
   }
   
   public static double[] getZPrime(double[] z,double[][] w){
      
      double[] a = new double[w.length];
      for(int r = 0; r<w.length; r++){
         for(int c = 0;c<w[0].length;c++){
            a[r] = a[r] + z[c] * w[r][c];
         }
      }
      
      return a;
   }
   

   
   public static double MSE(double[] fx,double[] y){//Mean Squared Error cost function
      
      double[] squareError = new double[fx.length];
      for(int i = 0;i<fx.length;i++){
         squareError[i] = Math.pow(fx[i]-y[i],2); 
      }
      
      double out = 0;
      for(double n:squareError){
         out = out + n;
      }
      
      return out / squareError.length;
   }
   
   public static double[] MSEPrime(double[] fx,double[] y){
      double[] out = new double[fx.length];
      for(int i = 0;i<fx.length;i++){
         out[i] = 2*(fx[i]-y[i])/fx.length;
      }
      
      return out;
   }
      
   //Feedforward code
   public double[] feedforward(double[] in){
      double[] a = new double[in.length];
      System.arraycopy(in,0,a,0,in.length);
      
      for(int i = 0;i<weights.length-1;i++){
         double[] z = getZ(a,weights[i],biases[i]);
         
         a = new double[z.length];
         for(int j = 0;j<z.length;j++){
            a[j] = relu(z[j]);
         }
      }
      
      double[] z = getZ(a,weights[weights.length-1],biases[biases.length-1]);
      a = new double[z.length];
      for(int j = 0;j<z.length;j++){
         a[j] = sigmoid(z[j]);
      }
      
      return a;
   }
   
   //Backprop code
   public double[][][] backprop(double[] x,double[] y){
      
      //Forward pass
      double[][] activations = new double[biases.length+1][];
      double[][] zs = new double[weights.length][];
      
      activations[0] = new double[x.length];
      System.arraycopy(x,0,activations[0],0,x.length);
      
      for(int i = 0;i<weights.length-1;i++){
         zs[i] = getZ(activations[i],weights[i],biases[i]);
            
         activations[i+1] = new double[zs[i].length];
         for(int j = 0;j<zs[i].length;j++){
            activations[i+1][j] = relu(zs[i][j]);
         }
      }
      
      zs[zs.length-1] = getZ(activations[activations.length-2],weights[weights.length-1],biases[biases.length-1]);
      activations[activations.length-1] = new double[zs[zs.length-1].length];
      for(int j = 0;j<zs[zs.length-1].length;j++){
         activations[activations.length-1][j] = sigmoid(zs[zs.length-1][j]);
      }

      
      //Backward Pass, obtain dC/dZ
      double[][] gradVector = new double[weights.length][];
      
      double[] dcdy = MSEPrime(activations[activations.length-1],y);
      
      gradVector[gradVector.length-1] = new double[dcdy.length];
      for(int i = 0;i<dcdy.length;i++){
         gradVector[gradVector.length-1][i] = sigmoidPrime(zs[zs.length-1][i]) * dcdy[i];
      }   
      
      for(int i = gradVector.length-2;i>=0;i--){
         double[] dcda = getZPrime(gradVector[i+1],weights[i+1]);
         gradVector[i] = new double[dcda.length];
         for(int j = 0;j>dcda.length;j++){
            gradVector[i][j] = dcda[j] * reluPrime(zs[i][j]);
         }
      }
      
      //Calc gradient vector in the format [w0,w1, ... wn,b]
      double[][][] grads = new double[gradVector.length+1][][];

      for(int l = 0;l<weights.length;l++){
         grads[l] = new double[weights[l].length][];
         
         for(int r = 0;r<weights[l].length;r++){
            grads[l][r] = new double[weights[l][r].length];
            
            for(int c = 0;c<weights[l][r].length;c++){
               grads[l][r][c] = activations[l][r] * gradVector[l][c];
               
            }
         }
      }
      
      grads[grads.length-1] = gradVector;
      
      
      return grads; 
   }
   
   public void sgd(double[] x,double[] y,double lr){//For applying stochatic gradient decent given x,y
      double[][][] dcdw = backprop(x,y); //1 longer because contains biases array, remember to iterate via weights.length
      double[][] dcdb = dcdw[dcdw.length-1];
      
      for(int i = 0; i < biases.length;i++){
         for(int c = 0; c < biases[i].length;c++){
            biases[i][c] = biases[i][c] - (dcdb[i][c] * lr);
            
            for(int r = 0; r < weights[i].length;r++){
               weights[i][r][c] = weights[i][r][c] - (dcdw[i][r][c] * lr); 
            }
         }
      }
   }
   
   public void upd(double[][][] dcdw,double lr){//for updating given grad vector in form w1,w2,wn,b
      double[][] dcdb = dcdw[dcdw.length-1];
      
      for(int i = 0; i < biases.length;i++){
         for(int c = 0; c < biases[i].length;c++){
            biases[i][c] = biases[i][c] - (dcdb[i][c] * lr);
            
            for(int r = 0; r < weights[i].length;r++){
               weights[i][r][c] = weights[i][r][c] - (dcdw[i][r][c] * lr); 
            }
         }
      }
   }
   
   public void write(String path){
      try {
         FileOutputStream fileOut = new FileOutputStream(path);
         ObjectOutputStream out = new ObjectOutputStream(fileOut);
         out.writeObject(this);
         out.close();
         fileOut.close();
      } catch (IOException i) {
         i.printStackTrace();
      }
   }
   
   public String toString(){
      String out = "";
      
      for(int i = 0;i<weights.length;i++){
         out = out + "Layer " + i + "\n"+"[\n";
         for(double[] a:weights[i]){
            out = out + Arrays.toString(a) + "\n";
         }
         out = out + "]\n" +Arrays.toString(biases[i])+"\n";
      }
      
      return out;
   }
   
}