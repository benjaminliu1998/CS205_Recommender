# CS205_Recommender

## step-by-step guide for running Python
1. From the GitHub repository, copy over the python script to the EMR cluster

   ```$ scp -i ~/.ssh/your_.pem_file_here python/recommender.py  hadoop@y*our_Master_public_DNS_here*:/home/hadoop```
   
2. Log in to the EMR cluster again

   ```$ ssh -i ~/.ssh/your_.pem_file_here hadoop@*your_Master_public_DNS_here*```
   
3. Now, upload the MovieLens dataset you want to use to the EMR cluster; for this example, we will upload the Movielens 20mL dataset

   1. If uploading the dataset from the public S3 bucket to the EMR cluster home repository
    
      ``` $ aws s3 cp s3://als-recommender-data/data/ratings_20ml.csv .```
       
   2. If uploading from the GitHub repository
   
       ```$ scp -i ~/.ssh/your_.pem_file_here data/mldataset.csv  hadoop@y*our_Master_public_DNS_here*:/home/hadoop```
       
4. Upload the dataset 'ratings_20ml.csv' to the Hadoop file system

   When running the command ```$ hadoop fs -ls```, you should see something similar to this: 
   
   ![Screen Shot 2021-05-09 at 3 50 19 PM](https://user-images.githubusercontent.com/37121874/117585088-8bd2ac00-b0de-11eb-9bfa-f1d05b9c609a.png)

5. You should now be able to run the below code and see results

      ``` spark-submit recommender.py ratings_20ml.csv ```
      
6. When the code has completed, you should be able to see the Mean Squared Error produced by the ALS PySpark Recommender

   ![Screen Shot 2021-05-09 at 3 56 13 PM](https://user-images.githubusercontent.com/37121874/117585237-59757e80-b0df-11eb-88d2-a3bd1ddd7704.png)

7. To profile the code and calculate execution time, from the **Summary** tab of your EMR cluster, click on *YARN timeline server* under the *Application user interfaces* section

   <img width="427" alt="Screen Shot 2021-05-09 at 4 03 47 PM" src="https://user-images.githubusercontent.com/37121874/117585409-5038e180-b0e0-11eb-927c-522e27f5157d.png">
   
8. You can now calculate the execution time of the recommender system. We see that the script took 10 minutes 17 seconds to run (StartTime: Sat May 8 12:17:23 - FinishTime: Sat May 8 12:27:40). To profile the code, you can click on the *History* link under the *Tracking UI* column header.

   <img width="1420" alt="Screen Shot 2021-05-09 at 4 10 15 PM" src="https://user-images.githubusercontent.com/37121874/117585589-45cb1780-b0e1-11eb-9f7c-0041a9b4654a.png">

    

