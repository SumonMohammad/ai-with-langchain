from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/c/N4XyA")
res= chain.invoke({"text": "Hello, world!", "language": "French"})
print(res)





