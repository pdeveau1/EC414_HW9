%https://www.mathworks.com/help/deeplearning/ref/patternnet.html;jsessionid=17e6dc5482df9e5be18de84c07e1
clear
%load the training data
[x,t] = iris_dataset;
%construct a pattern network with one hidden layer of size 10
net = patternnet(10);
%train the network net using the training data
net = train(net,x,t);
%view the trained network
view(net)
%estimate the targets using the trained network
y = net(x);
%assess the performance of the trained network
perf = perform(net,t,y)
classes = vec2ind(y);