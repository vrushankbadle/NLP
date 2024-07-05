from website import create_app 

port = 5000

app = create_app()

if __name__ == '__main__':
    app.run(port=port)

