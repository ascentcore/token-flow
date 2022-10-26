import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
// import 'react-tabs/style/react-tabs.css';
import m1 from '/public/drive.png';
import m2 from '/public/narnia.png';
import m3 from '/public/basic.png';

export default () => (
  <Tabs>
    <TabList>
      <Tab>Largt Models</Tab>
      <Tab>Behavior</Tab>
      <Tab>Our Approach</Tab>
    </TabList>

    <TabPanel>
      <div class="image-slide">
        <div class="image-container">
          <img src={m1} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div class="image-slide">
        <div class="image-container">
          <img src={m2} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div class="image-slide">
        <div class="image-container" style={{marginTop: 20}}>
          <img src={m3} alt="Logo" />
        </div>
      </div>
    </TabPanel>
   
  </Tabs>
);
