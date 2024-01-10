docker-code-lint:
	echo "Checking Codestyle"
	docker run --rm -v ${CURDIR}:/data cytopia/black --check .
