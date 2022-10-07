import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
// import 'react-tabs/style/react-tabs.css';
import swdev from '/public/swdev.png'; 
import cto from '/public/cto.png'; 
import research from '/public/lead-researcher.png'; 
import mu from '/public/mu.png'; 

export default () => (
  <Tabs>
    <TabList>
      <Tab>Software Developer</Tab>
      <Tab>CTO@AC</Tab>
      <Tab>ResearchLead@AC</Tab>
      <Tab>Monkeyuser</Tab>
    </TabList>

    <TabPanel>
        <div class="image-container">
            <img src={swdev} alt="Logo" />
        </div>
    </TabPanel>
    <TabPanel>
        <div class="image-container">
            <img src={cto} alt="Logo" />
        </div>
    </TabPanel>
    <TabPanel>
        <div class="image-container">
            <img src={research} alt="Logo" />
        </div>
    </TabPanel>
    <TabPanel>
        <div class="image-container">
            <img src={mu} alt="Logo" />
        </div>
    </TabPanel>
    
  </Tabs>
);
