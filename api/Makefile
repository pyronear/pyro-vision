# Pin the dependencies
lock:
	poetry lock
	poetry export -f requirements.txt --without-hashes --output requirements.txt
	poetry export -f requirements.txt --without-hashes --with dev --output requirements-dev.txt

# Run the docker
run:
	docker-compose up -d --build

# Run the docker
stop:
	docker-compose down

# Run tests for the library
test:
	docker-compose up -d --build
	docker cp requirements-dev.txt api_backend_1:/app/requirements-dev.txt
	docker-compose exec -T backend pip install -r requirements-dev.txt
	docker cp tests api_backend_1:/app/tests
	docker-compose exec -T backend pytest tests/
	docker-compose down
