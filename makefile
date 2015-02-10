default:
	g++ lab1.cpp -o lab1 `pkg-config opencv --cflags --libs`
clean:
	rm lab1
