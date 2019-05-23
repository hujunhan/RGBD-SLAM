//
// Created by hu on 2019/5/23.
//
#include "yaml-cpp/yaml.h"
#include "iostream"
using namespace std;
int main()
{
    YAML::Node config = YAML::LoadFile("../config.yaml");
    cout<<config["number"].as<int>()+2<<endl;
    cout<<config["string"]<<endl;
    return 0;
}