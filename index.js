const fs = require('fs');
const moment = require('moment');
const simpleGit = require('simple-git');
const events = require('events');

const FILE_PATH = './data.json';
const git = simpleGit();

function generateRandomDate() {
  // Subtract a random number of weeks from a year ago
  const baseDate = moment().subtract(1, 'year');

  // Generate random values for weeks and days within their respective ranges
  const randomWeeks = Math.floor(Math.random() * 52); // Adjust for non-leap years if desired
  const randomDays = Math.floor(Math.random() * 7);


  const randomHour = Math.floor(Math.random() * 24);
  const randomMinute = Math.floor(Math.random() * 60);
  const randomSecond = Math.floor(Math.random() * 60);
  const randomDate = baseDate
  .add(randomWeeks, 'weeks')
  .add(randomDays, 'days')
  .set('hour', randomHour)
  .set('minute', randomMinute)
  .set('second', randomSecond);
  return randomDate.format('YYYY-MM-DD HH:mm:ss');
}


async function main() {
  const randomModule = await import('random');
  const random = randomModule.default; // Adjust based on the actual export from the 'random' module

  const makeCommit = async (n) => {
    if (n === 0) {
      return git.push();
    }

    const DATE = generateRandomDate();
    const data = { date: DATE };

    console.log(`Attempting to commit for date: ${DATE}`);

    try {
      await fs.promises.writeFile(FILE_PATH, JSON.stringify(data));
      await git.add([FILE_PATH]);
      await git.commit(`Commit for ${DATE}`, { '--date': DATE });
      await git.push(['origin', 'main'], { '--verbose': null, '--porcelain': null });
      await makeCommit(n - 1); // Await recursive call
    } catch (error) {
      console.error('Error making commit:', error);
      return; // Stop further recursion on error
    }
  };

  await makeCommit(10);
}

main();