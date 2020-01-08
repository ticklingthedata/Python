# Selenium: A quick recap

## Navigating:

* Interacting with a page:
	
	* Finding elements: 
	
	  * ```python
	    element = driver.find_element_by_id("passwd-id")
	    ```
	
	  * If not found: *NoSuchElementException*
	
	* Filling fields:
	
	  * ```python
	    element.send_keys("some text")
	    ```
	
	* Clearing fields:
	
	  * ```python
	    element.clear()
	    ```
	
* Filling forms:

  * Selecting/deselecting elements:

    * ```python
      select = Select(driver.find_element_by_name('name'))
      select.select_by_visible_text("text")
      select.deselect_all()
      ```

  * Getting selected:

    * ```python
      select = Select(driver.find_element_by_xpath("//select[@name='name']"))
      all_selected_options = select.all_selected_options
      ```

  * Submitting:

    * ```python
      driver.find_element_by_id("submit").click()
      ```

* Drag and drop:

  * ```python
    element = driver.find_element_by_name("source")
    target = driver.find_element_by_name("target")
    
    from selenium.webdriver import ActionChains
    action_chains = ActionChains(driver)
    action_chains.drag_and_drop(element, target).perform()
    ```

* Moving between windows/frames:

  * ```python
    driver.switch_to_window("windowName")
    ```

* Pop-ups:

  * ```python
    alert = driver.switch_to_alert()
    ```

* History and location:

  * ```python
    driver.get("http://www.example.com")
    driver.forward()
    driver.back()
    ```

* Cookies:

  * ```python
    driver.get("http://www.example.com")
    
    # Now set the cookie. This one's valid for the entire domain
    cookie = {‘name’ : ‘foo’, ‘value’ : ‘bar’}
    driver.add_cookie(cookie)
    
    # And now output all the available cookies for the current URL
    driver.get_cookies()
    ```



## Locating:

* Finding elements:

  * Finding one element:

  ```markdown
  find_element_by_id
  find_element_by_name
  find_element_by_xpath
  find_element_by_link_text
  find_element_by_partial_link_text
  find_element_by_tag_name
  find_element_by_class_name
  find_element_by_css_selector
  ```

  * Finding multiple elements: retuns a list of elements

  ```markdown
  find_elements_by_name
  find_elements_by_xpath
  find_elements_by_link_text
  find_elements_by_partial_link_text
  find_elements_by_tag_name
  find_elements_by_class_name
  find_elements_by_css_selector
  ```

* Locating by Id:

  * ```python
    login_form = driver.find_element_by_id('loginForm')
    ```

* Locating by name:

  * ```python
    username = driver.find_element_by_name('username')
    password = driver.find_element_by_name('password')
    ```

* Locating by XPath:

  * ```python
    username = driver.find_element_by_xpath("//input[@name='username']")
    ```

* Locating hyperlinks:

  * ```python
    continue_link = driver.find_element_by_link_text('Continue')
    continue_link = driver.find_element_by_partial_link_text('Conti')
    ```

* Locating by Tag Name:

  * ```python
    heading1 = driver.find_element_by_tag_name('h1')
    ```

* Locating by Class Name:

  * ```python
    content = driver.find_element_by_class_name('content')
    ```

* Locating by CSS Selectors:

  * ```python
    content = driver.find_element_by_css_selector('p.content')
    ```



## Waits:

- Explicit waits:

  - An explicit wait is a code you define to wait for a certain condition to occur before proceeding further in the code. (*ExpectedCondition*)

  - ```python
    from selenium.webdriver.support import expected_conditions as EC
    
    driver = webdriver.Firefox()
    driver.get("http://somedomain/url_that_delays_loading")
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "myDynamicElement"))
        )
    finally:
        driver.quit()
    ```

  - Expected conditions:

    - ```markdown
      title_is
      title_contains
      presence_of_element_located
      visibility_of_element_located
      visibility_of
      presence_of_all_elements_located
      text_to_be_present_in_element
      text_to_be_present_in_element_value
      frame_to_be_available_and_switch_to_it
      invisibility_of_element_located
      element_to_be_clickable
      staleness_of
      element_to_be_selected
      element_located_to_be_selected
      element_selection_state_to_be
      element_located_selection_state_to_be
      alert_is_present
      ```

- Implicit waits:

  - An implicit wait tells WebDriver to poll the DOM for a certain amount of time when trying to find any element (or elements) not immediately available. The default setting is 0. Once set, the implicit wait is set for the life of the WebDriver object.

  - ```python
    from selenium import webdriver
    
    driver = webdriver.Firefox()
    driver.implicitly_wait(10) # seconds
    driver.get("http://somedomain/url_that_delays_loading")
    myDynamicElement = driver.find_element_by_id("myDynamicElement")
    ```



## Webdriver API:

- Exceptions (most common):

  - **ElementNotSelectableException**: Thrown when trying to select an unselectable element.

  - **ElementNotVisibleException**: Thrown when an element is present on the DOM, but it is not visible, and so is not able to be interacted with.

  - **NoSuchAttributeException**: Thrown when the attribute of element could not be found.

  - **NoSuchElementException**: Thrown when element could not be found.

  - **TimeoutException**: Thrown when a command does not complete in enough time.

    

- Action chains:

  - ActionChains are a way to automate low level interactions such as mouse movements, mouse button actions, key press, and context menu interactions. This is useful for doing more complex actions like hover over and drag and drop.

  - ```python
    menu = driver.find_element_by_css_selector(".nav")
    hidden_submenu = driver.find_element_by_css_selector(".nav #submenu1")
    
    actions = ActionChains(driver)
    actions.move_to_element(menu)
    actions.click(hidden_submenu)
    actions.perform()
    ```

  - Available actions (most common):

    - `click`(*on_element=None*)

    - `double_click`(*on_element=None*)

    - `drag_and_drop`(*source*, *target*)

    - `move_to_element`(*to_element*)

    - `perform()`: Performs all stored actions.

    - `send_keys`(**keys_to_send*)

      

- Alerts:

  - Used to interact with alert prompts. It contains methods for dismissing, accepting, inputting, and getting text from alert prompts.
  - Available methods (most common):
    - `accept`()
    - `dismiss`()
    - `send_keys`(*keysToSend*)