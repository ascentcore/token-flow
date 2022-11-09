import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
// import 'react-tabs/style/react-tabs.css';
import m1 from '/public/drive.png';
import m2 from '/public/narnia.png';
import m3 from '/public/basic.png';
import m4 from '/public/context.mp4';
import m5 from '/public/meaning-of-life.png';
import m6 from '/public/messages.png';

export default () => (
  <Tabs>
    <TabList>
      <Tab>Large Models</Tab>
      <Tab>Behavior</Tab>
      <Tab>Idea</Tab>
      <Tab>Approach</Tab>
      <Tab>Use Case #1</Tab>
      <Tab>Use Case #2</Tab>
    </TabList>

    <TabPanel>
      <div className="image-slide">
        <div className="image-container">
          <img src={m1} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container">
          <img src={m2} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container" style={{marginTop: 20}}>
          <img src={m3} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container" style={{marginTop: 20}}>
        <video width="600" height="600" controls>
          <source src={m4} type="video/mp4"/>
        </video>
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container" style={{marginTop: 20}}>
          <img src={m5} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container" style={{marginTop: 20}}>
          <img src={m6} alt="Logo" />
        </div>
      </div>
    </TabPanel>
  </Tabs>
);
