"""
The idea is simple -> taking two arrays of shape (9,9) as the input/output, first will be the X_train, second the y_train.

    
           X_train                              y_train

    ([0,9,0 ,0,0,6 ,1,0,0],              ([4,9,2 ,7,8,6 ,1,3,5],          
     [7,1,0 ,0,9,0 ,0,2,8],               [7,1,3 ,5,9,4 ,6,2,8],        
     [0,0,8 ,0,0,0 ,0,4,9],               [6,5,8 ,3,1,2 ,7,4,9],        
                                                                                  
     [0,6,0 ,2,5,0 ,0,0,1],               [8,6,7 ,2,5,3 ,4,9,1],           
     [0,0,0 ,0,0,0 ,0,0,0],               [2,3,9 ,4,7,1 ,5,8,6],           
     [5,0,0 ,0,6,9 ,0,7,0],               [5,4,1 ,8,6,9 ,2,7,3],           
                                                                  
     [1,2,0 ,0,0,0 ,8,0,0],               [1,2,4 ,9,3,5 ,8,6,7],           
     [9,8,0 ,0,4,0 ,0,5,2],               [9,8,6 ,1,4,7 ,3,5,2],           
     [0,0,5 ,6,0,0 ,0,1,0])               [3,7,5 ,6,2,8 ,9,1,4])  



Wondering only how many of these arrays i need to feed the model to get working ,.. ?
Depending of the batches,... epochs,... starting with 50 arrays,..... hmm ?
Tomorrow we'll see.

"""

X_train = ( # 10 easy sudoku 
        ([9,0,0,8,0,0,0,1,0],[0,8,0,0,1,0,0,7,5],[0,0,5,0,0,2,0,0,0],[4,0,0,0,0,0,3,0,0],[0,3,0,0,2,0,0,5,0],[0,0,6,0,0,0,0,0,4],[0,0,0,3,0,0,4,0,0],[3,2,0,0,4,0,0,9,0],[0,5,0,0,0,1,0,0,7]),
        ([0,0,5,0,0,9,6,0,0],[0,7,0,0,4,0,0,8,0],[8,0,0,0,0,0,0,0,5],[9,0,0,1,0,4,0,0,0],[0,2,0,0,0,0,0,5,0],[0,0,0,7,0,8,0,0,4],[5,0,0,0,0,0,0,0,1],[0,9,0,0,6,0,0,2,0],[0,0,1,9,0,0,3,0,0]),
        ([0,0,0,6,0,0,8,0,0],[0,0,6,0,0,9,0,0,0],[0,7,0,0,8,0,0,0,5],[1,0,0,5,0,7,0,4,0],[0,0,7,0,0,0,3,0,0],[0,2,0,1,0,6,0,0,7],[4,0,0,0,5,0,0,9,0],[0,0,0,4,0,0,2,0,0],[0,0,9,0,0,3,0,0,0]),
        ([0,5,0,7,0,0,0,1,0],[3,0,6,0,5,0,0,0,2],[0,0,0,9,0,0,0,6,0],[0,0,0,0,0,0,6,0,1],[0,6,0,0,7,0,0,5,0],[4,0,8,0,0,0,0,0,0],[0,1,0,0,0,7,0,0,0],[7,0,0,0,3,0,2,0,5],[0,8,0,0,0,2,0,7,0]),
        ([9,0,0,2,0,0,0,6,7],[7,0,0,0,0,5,8,0,0],[0,2,0,0,1,0,0,0,0],[0,9,0,0,0,0,0,0,2],[0,0,8,0,2,0,5,0,0],[2,0,0,0,0,0,0,4,0],[0,0,0,0,8,0,0,1,0],[0,0,9,5,0,0,0,0,4],[4,3,0,0,0,7,0,0,5]),
        ([9,0,0,8,0,7,0,0,0],[0,6,0,0,0,0,2,0,3],[0,0,0,0,0,0,0,0,0],[0,2,0,0,0,5,0,0,6],[0,0,0,0,6,0,0,0,0],[8,0,0,9,0,0,0,7,0],[0,0,0,0,0,0,0,0,0],[4,0,9,0,0,0,0,8,0],[0,0,0,3,0,6,0,0,2]),
        ([0,0,8,0,0,0,0,6,0],[0,0,0,6,0,0,0,0,0],[0,1,0,0,3,0,0,0,7],[0,0,0,0,0,2,1,0,0],[2,0,1,9,0,7,4,0,8],[0,0,5,8,0,0,0,0,0],[3,0,0,0,6,0,0,9,0],[0,0,0,0,0,9,0,0,0],[0,4,0,0,0,0,7,0,0]),
        ([2,0,0,0,0,7,0,0,0],[0,5,0,0,4,0,0,7,2],[0,0,7,3,0,0,1,0,0],[0,0,0,0,0,0,6,0,0],[1,8,0,0,2,0,0,4,7],[0,0,9,0,0,0,0,0,0],[0,0,2,0,0,3,4,0,0],[8,6,0,0,5,0,0,1,0],[0,0,0,1,0,0,0,0,6]),
        ([6,0,0,0,0,4,0,0,0],[0,8,0,0,6,0,0,0,1],[0,0,9,2,0,0,8,7,0],[0,0,0,0,0,0,1,4,0],[7,0,0,0,2,0,0,0,5],[0,6,3,0,0,0,0,0,0],[0,5,6,0,0,1,3,0,0],[3,0,0,0,8,0,0,1,0],[0,0,0,9,0,0,0,0,2]),
        ([0,0,0,0,0,0,0,0,0],[7,0,0,0,3,0,0,9,0],[0,4,0,0,6,0,5,0,0],[0,0,0,9,0,0,4,0,0],[8,0,0,0,1,0,0,0,6],[0,0,9,0,0,3,0,0,0],[0,0,1,0,8,0,0,2,0],[0,5,0,0,4,0,0,0,7],[0,0,0,0,0,0,0,0,0]),

        # 10 medium sudoku
       
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),


        # 10 hard sudoku
   
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
)


y_train = ( # 10 easy sudoku 
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),

        # 10 medium sudoku
       
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),


        # 10 hard sudoku
   
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
        ([],[],[],[],[],[],[],[],[]),
)
