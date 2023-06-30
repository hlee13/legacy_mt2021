  int N = 5;
  for (int i=0, j=0; i<N; i=j+1<N?i:i+1, j++, j%=N) {
      System.out.println(i + " " + j);
  }

  // for (int i=0, j=0; i<N; j=j<i?j+1:0, i=j>0?i:i+1) {
  for (int i=0, j=0; i<N; j=(++j)%(i+1), i=j>0?i:i+1) {
      System.out.println(i + " " + j);
  }
