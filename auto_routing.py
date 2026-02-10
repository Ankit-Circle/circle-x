from app.auto_routing_app import create_auto_routing_app

app = create_auto_routing_app()

if __name__ == "__main__":
    app.run(debug=True)
