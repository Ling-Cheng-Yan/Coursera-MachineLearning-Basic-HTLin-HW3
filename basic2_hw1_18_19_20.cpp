#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
 
using namespace std;
 
#define DEMENSION 20//数据维度
 
struct Record{  //数据格式
	double x[DEMENSION+1];
	int y;
};
 
struct Weight{  //参数格式
	double w[DEMENSION+1];
};
 
int sign(double x){  //sign
	if(x > 0)return 1;
	else return -1;
}
 
void getData(fstream &datafile,vector<Record> &data){  //读取数据
	while(!datafile.eof()){
		Record temp;
		temp.x[0] = 1;
		for(int i = 1; i <= DEMENSION; i++)
			datafile>>temp.x[i];
		datafile>>temp.y;
		data.push_back(temp);
	}
	datafile.close();
}
 
double sigmoid(double x){  //sigmoid函数，逻辑函数，s形函数
	return 1.0 / (1.0 + exp(-x));
}
 
double vectorMul(double *a,double *b,int demension){ //两个向量相乘返回内积
	double temp = 0.0;
	for(int i = 0; i <demension; i++)
		temp += a[i] * b[i];
	return temp;
}
 
void calcuBatchGradient(vector<Record> &data,Weight weight,int N,double *grad){  //批量梯度下降法
	for(int i = 0; i < N; i++){
		double temp = sigmoid(-1 * vectorMul(weight.w,data[i].x,DEMENSION+1) * (double)data[i].y);
		for(int j = 0; j <= DEMENSION; j++)
			grad[j] += -1.0 * temp * data[i].x[j] * data[i].y; 
	}
	for(int i = 0; i <= DEMENSION; i++)
		grad[i] = grad[i] / N;
}
 
void calcuStochasticGradient(Record data,Weight weight,double *grad){  //随机梯度下降法
	double temp = sigmoid(-1 * vectorMul(weight.w,data.x,DEMENSION+1) * (double)data.y);
	for(int j = 0; j <= DEMENSION; j++)
		grad[j] += -1.0 * temp * data.x[j] * data.y;
 
}
 
void updateW(Weight &weight,double ita,double *grad){  //利用得到的梯度更新参数weight
	for(int i = 0; i <= DEMENSION; i++){
		weight.w[i] = weight.w[i] - (ita * grad[i]);
	}
}
 
double calcuLGError(vector<Record> &data,Weight weight,int N){ //计算逻辑回归的错误计算方法计算错误
	double error = 0.0;
	for(int i = 0; i < N; i++){
		error += log(1 + exp(-data[i].y * vectorMul(weight.w,data[i].x,DEMENSION+1)));
	}
	return double(error / N);
}
 
void logisticRegression(vector<Record> &data,Weight &weight,int N,double ita,int iteration){  //逻辑回归
    for(int i = 0; i < iteration; i++){     //利用batch梯度下降法计算逻辑回归
		double grad[DEMENSION+1] = {0.0};
		calcuBatchGradient(data,weight,N,grad);
		updateW(weight,ita,grad);
		cout<<"iter = "<<i<<"Ein = "<<calcuLGError(data,weight,N)<<endl;
	}
	/*int i = 0;   //利用Stochastic梯度下降法计算逻辑回归
	while(i < iteration){
		double grad[DEMENSION+1] = {0.0};
		calcuStochasticGradient(data[i%N],weight,grad);
		updateW(weight,ita,grad);
		cout<<"iter = "<<i<<",训练样本的逻辑回归错误Ein = "<<calcuLGError(data,weight,N)<<endl;
		i++;
	}*/
}
 
double calcuError(vector<Record> &data,Weight weight,int N){  //利用逻辑回归做二元分类，计算0/1错误
	double error = 0.0;
	for(int i = 0; i < N; i++){
		if(sign(vectorMul(data[i].x,weight.w,DEMENSION+1)) != data[i].y)
			error++;
	}
	return double(error / N);
}
 
int main(){
	vector<Record> trainingData;  //训练样本
	vector<Record> testData;      //测试样本
	fstream file1("hw3_train.dat.txt");//读取训练样本数据
	fstream file2("hw3_test.dat.txt");//读取测试样本数据
	if(file1.is_open() && file2.is_open()){
		getData(file1,trainingData);
		getData(file2,testData);
	}
	else{
		cout<<"can not open file!"<<endl;
		exit(1);
	}
	int train_N = trainingData.size();//训练样本个数
	int test_N = testData.size();//测试样本个数
	double ita = 0.001;//步长ita
	int interation = 2000;//迭代次数
	Weight weight;//逻辑回归参数
	for(int i = 0; i <= DEMENSION; i++)//参数初始化为0；注意，这里要是全为1迭代2000次是得不到结果的，因为最优解在0附近；要想得到结果iteration必须在几万次次左右
		weight.w[i] = 1;
	logisticRegression(trainingData,weight,train_N,ita,interation);
	cout<<"TrainData="<<calcuError(trainingData,weight,train_N)<<endl;
    cout<<"TestData="<<calcuError(testData,weight,test_N)<<endl;
}