# Test specification: Flinks challenge

## Scenario: End to end test (auth, assert, filter and present)

* For each test case, get RequestId using the following connection information

| instance    |            customerId                | institution   |         username         | password | expectedStatusCode |
| ----------- | ------------------------------------ | ------------- | ------------------------ | -------- | ------------------ |
| toolbox-api | 43387ca6-0391-4c82-857d-70d95f087ecb | FlinksCapital |         Greatday         | Everyday |        203         |
| toolbox-api | 43387ca6-0391-4c82-857d-70d95f087ecb | FlinksCapital |   test_disabled_account  | Everyday |        401         |
| toolbox-api | 43387ca6-0391-4c82-857d-70d95f087ecb | FlinksCapital | test_service_unavailable | Everyday |        401         |
| toolbox-api | 43387ca6-0391-4c82-857d-70d95f087ecb | FlinksCapital |      test_dispatched     | Everyday |        203         |
| toolbox-api | 43387ca6-0391-4c82-857d-70d95f087ecb | FlinksCapital |       test_pending       | Everyday |        203         |


* Authorize each RequestId (!=401) using the following challenges:

|             Question               |                 Answer               |
| ---------------------------------- | ------------------------------------ |
| What city were you born in?        |                Montreal              |
| What is the best country on earth? |                Canada                |
| What shape do people like most?    |                Triangle              |

* Get the accounts detail for each authorized request

* Filter the details and present the results in the desired format


