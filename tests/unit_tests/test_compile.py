import os
import base64
from react_agent.tools import compile_component, take_screenshots
from react_agent.utils import FileContent

# Create a simple React component for testing
test_component = [
    FileContent(
        filename="index.tsx",
        content="""
import React from 'react';
import { Usage } from './Usage';

const exampleData = {
    workspaces: ['Workspace 1', 'Workspace 2', 'Workspace 3'],
    apiKeys: ['key_123', 'key_456', 'key_789'],
    models: ['GPT-4', 'GPT-3.5', 'Claude'],
    currentDate: '2024-03',
    tokensIn: 1234567,
    tokensOut: 987654,
    usageData: [
        { date: '2024-03-01', tokens: 150 },
        { date: '2024-03-02', tokens: 120 },
        { date: '2024-03-03', tokens: 180 },
        { date: '2024-03-04', tokens: 90 },
        { date: '2024-03-05', tokens: 200 },
    ]
};

const App = () => {
    return (
        <Usage {...exampleData} />
    );
};

export default App;
""",
        file_type="tsx",
        entrypoint=True
    ),
    FileContent(
        filename="Usage.tsx",
        content="import React from 'react'; import { UsageHeader } from './UsageHeader'; import { TokenStats } from './TokenStats'; import { UsageChart } from './UsageChart'; import './Usage.css'; export interface UsageData { date: string; tokens: number; } interface UsageProps { workspaces: string[]; apiKeys: string[]; models: string[]; currentDate: string; tokensIn: number; tokensOut: number; usageData: UsageData[]; } export const Usage: React.FC<UsageProps> = ({ workspaces, apiKeys, models, currentDate, tokensIn, tokensOut, usageData }) => { return ( <div className='usage'> <h1 className='usage__title'>Usage</h1> <UsageHeader workspaces={workspaces} apiKeys={apiKeys} models={models} currentDate={currentDate} /> <div className='usage__stats'> <TokenStats title='Total tokens in' value={tokensIn} /> <TokenStats title='Total tokens out' value={tokensOut} /> </div> <div className='usage__chart-container'> <h2 className='usage__chart-title'>Daily token usage</h2> <p className='usage__chart-subtitle'>Includes usage from both API and Console</p> <UsageChart data={usageData} /> </div> </div> ); };",
        file_type="tsx",
        entrypoint=False
    ),
    FileContent(
        filename="UsageHeader.tsx",
        content="import React from 'react'; import './Usage.css'; interface UsageHeaderProps { workspaces: string[]; apiKeys: string[]; models: string[]; currentDate: string; } export const UsageHeader: React.FC<UsageHeaderProps> = ({ workspaces, apiKeys, models, currentDate }) => { return ( <div className=\"usage-header\"> <div className=\"usage-header__filters\"> <select className=\"usage-header__select\"> <option>All Workspaces</option> {workspaces.map(workspace => ( <option key={workspace}>{workspace}</option> ))} </select> <select className=\"usage-header__select\"> <option>All API keys</option> {apiKeys.map(key => ( <option key={key}>{key}</option> ))} </select> <select className=\"usage-header__select\"> <option>All Models</option> {models.map(model => ( <option key={model}>{model}</option> ))} </select> </div> <div className=\"usage-header__controls\"> <div className=\"usage-header__date-picker\"> <button className=\"usage-header__arrow\">←</button> <span className=\"usage-header__date\">{currentDate}</span> <button className=\"usage-header__arrow\">→</button> </div> <div className=\"usage-header__group\"> <select className=\"usage-header__select\"> <option>Group by: None</option> </select> </div> <button className=\"usage-header__export\">Export</button> </div> </div> ); };",
        file_type="tsx",
        entrypoint=False
    ),
    FileContent(
        filename="TokenStats.tsx",
        content="import React from 'react'; import './Usage.css'; interface TokenStatsProps { title: string; value: number; } export const TokenStats: React.FC<TokenStatsProps> = ({ title, value }) => { return ( <div className=\"token-stats\"> <h2 className=\"token-stats__title\">{title}</h2> <div className=\"token-stats__value\">{value}</div> </div> ); };",
        file_type="tsx",
        entrypoint=False
    ),
    FileContent(
        filename="UsageChart.tsx",
        content="import React from 'react'; import { UsageData } from './Usage'; import './Usage.css'; interface UsageChartProps { data: UsageData[]; } export const UsageChart: React.FC<UsageChartProps> = ({ data }) => { const maxValue = Math.max(...data.map(d => d.tokens)); return ( <div className=\"usage-chart\"> <div className=\"usage-chart__y-axis\"> {[0, 50, 100, 150, 200].map(value => ( <div key={value} className=\"usage-chart__y-label\">{value}</div> ))} </div> <div className=\"usage-chart__bars\"> {data.map((item) => ( <div key={item.date} className=\"usage-chart__bar-container\"> <div className=\"usage-chart__bar\" style={{ height: `${(item.tokens / maxValue) * 100}%` }} /> <div className=\"usage-chart__x-label\">{item.date}</div> </div> ))} </div> </div> ); };",
        file_type="tsx",
        entrypoint=False
    ),
    FileContent(
        filename="Usage.css",
        content="/* Usage.css */ .usage { background-color: #1C1C1C; color: white; padding: 20px; } .usage__title { font-size: 24px; margin-bottom: 20px; } .usage-header { display: flex; justify-content: space-between; margin-bottom: 30px; } .usage-header__filters, .usage-header__controls { display: flex; gap: 10px; } .usage-header__select { background-color: #2C2C2C; color: white; border: none; padding: 8px 16px; border-radius: 4px; } .usage-header__date-picker { display: flex; align-items: center; gap: 10px; } .usage-header__arrow { background: none; border: none; color: white; cursor: pointer; } .usage-header__export { background-color: #B85C38; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; } .usage__stats { display: flex; gap: 20px; margin-bottom: 30px; } .token-stats { background-color: #2C2C2C; padding: 20px; border-radius: 8px; flex: 1; } .token-stats__title { font-size: 16px; margin-bottom: 10px; } .token-stats__value { font-size: 32px; font-weight: bold; } .usage__chart-container { background-color: #2C2C2C; padding: 20px; border-radius: 8px; } .usage__chart-title { font-size: 18px; margin-bottom: 5px; } .usage__chart-subtitle { color: #888; font-size: 14px; margin-bottom: 20px; } .usage-chart { display: flex; height: 300px; padding-bottom: 30px; } .usage-chart__y-axis { display: flex; flex-direction: column-reverse; justify-content: space-between; padding-right: 20px; } .usage-chart__y-label { color: #888; } .usage-chart__bars { display: flex; align-items: flex-end; gap: 40px; flex: 1; } .usage-chart__bar-container { display: flex; flex-direction: column; align-items: center; flex: 1; } .usage-chart__bar { width: 40px; background-color: #B85C38; transition: height 0.3s ease; } .usage-chart__x-label { margin-top: 10px; color: #888; }",
        file_type="css",
        entrypoint=False
    )
]

def save_screenshot(screenshot_data: str, filename: str):
    """Save base64 encoded screenshot to file"""
    # Create screenshots directory if it doesn't exist
    os.makedirs("screenshots", exist_ok=True)
    
    # Decode base64 and save to file
    image_data = base64.b64decode(screenshot_data)
    file_path = os.path.join("screenshots", filename)
    with open(file_path, "wb") as f:
        f.write(image_data)
    return file_path

def main():
    print("Compiling component...")
    result = compile_component(test_component, config={})
    
    if "errors" in result:
        print("Compilation failed!")
        print("Errors:", result["errors"])
        return

    print("Compilation successful!")
    print(f"Dev server URL: {result['dev_server_url']}")
    
    # Take screenshots
    print("\nTaking screenshots...")
    screenshots = take_screenshots(result['dev_server_url'], config={})
    
    if "errors" in screenshots:
        print("Screenshot capture failed!")
        print("Errors:", screenshots["errors"])
    else:
        # Save screenshots
        for view_type, screenshot_data in screenshots.items():
            file_path = save_screenshot(screenshot_data, f"{view_type}.png")
            print(f"Saved {view_type} screenshot to: {file_path}")
    
    print("\nThe component should now be visible in your browser.")
    print("Press Ctrl+C to stop the server when done.")
    
    # Keep the script running to maintain the server
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main() 