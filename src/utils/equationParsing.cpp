#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>

#define DEG 100	//for polynomials of max degree = 99

int main()
{
	static const size_t npos = -1;
	std::string str;
	int obfs;
	std::cout << "      Enter a polynomial without like terms\n";
	std::cout << "(use the letter x. for ex.: -x+4.5x^3+2x^4-3.1)\n";
	std::cout << "\nEnter: ";
	std::cin >> str;
	if(str == "") return 1;
	int strSize = str.size();
	std::cout << "Enter total number of objectives/criteria\n";
	std::cout << "\nEnter: ";
	std::cin >> obfs;
	
//	How many monomials has the polynomial?
	int k = 1;
	for(int i = 1; i < strSize; ++i)
		if(str[i] == '+' || str[i] == '-')
			k++;
 	int monoms = k ;
 	
//	Signs "+" are necessary for the string parsing 
	if(isdigit(str[0])) str.insert(0, "+");
	if(str[0] == 'x') str.insert(0, "+");
	str.append("+");
	strSize = str.size();
	
//	Extracting the monomials as monomStr
	k = 0;
	int j = 0;
	std::string monomStr[DEG];
	for(int i = 1; i < strSize; ++i)
		if(str[i] == '+' || str[i] == '-')
		{
			monomStr[k++] = str.substr(j, i - j);
			j = i;
		}


//  Monomials' formatting i.e. to have all the same form: coefficientX^exponent
	for(int i = 0; i < monoms; ++i)
	{
		if(monomStr[i][1] == 'x')	//x is after the +/- sign 
			monomStr[i].insert(1, "1");	//& gets 1 as coefficient
		bool flag = false;	// assuming that x is not present
		int len = monomStr[i].size();
		for(int j = 1; j < len; ++j)
			if(monomStr[i][j] == 'x')	//but we test this
			{
				flag = true;	//& if x is present
				if(j == len - 1)	//& is the last 
					monomStr[i].append("^1");	//it gets exponent 1
				break;	//& exit from j_loop
			}
		if(!flag)	//if x is not present: we have a constant term
			monomStr[i].append("x^0");	//who gets "formatting"
	}
	
	
	//extracting weights and degrees from monomStr
	std::vector <double> weights(monoms,1); //we have monoms weights

	
	// extraction of weights
	for (int i=0; i<monoms;i++){
		if (monomStr[i].find('x')==std::string::npos){			//because of some errors I had to replace string::npos with -1;
			weights[i]=stoi(monomStr[i]);
			continue;}
		for (int j=0; j<monomStr[i].size(); i++){
			if(monomStr[i][j] == 'x' || monomStr[i][j] == 'X'){
				if (j==1) 
				{ break;
				}
				else{
				weights[i]=stoi(monomStr[i].substr(0,j-1));
				break;
			}
		}
	}
}

	//extractiion of degress
	std::vector<std::vector<int>> degrees; 						//degrees has monoms elemets (for each monomial of the polonomial), each element is a vector of m elements 
	degrees.resize(monoms,std::vector<int> (obfs));		//where m is the number of obfs each of the m elements specify the degree of that obf in this monomial
	for (int i=0; i<monoms;i++){
	j=0;
	int xindex1, xindex2, pindex, o;
	while (j<monomStr[i].size()){
		if (monomStr[i].find('x')==std::string::npos){break;}
		xindex2=monomStr[i].find('x',xindex1+1);
		pindex=monomStr[i].find('^',xindex1+1);
		o=stoi(monomStr[i].substr(xindex1+1,std::min(xindex2,pindex)));
		degrees[i][o]=1; j=std::min(xindex2,pindex);
		if (xindex2>pindex ){
			degrees[i][o]=stoi(monomStr[i].substr(pindex+1,xindex2));
			j=xindex2;
		}
	}
}
			
	return 0;
}